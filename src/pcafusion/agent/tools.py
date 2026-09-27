"""Tools exposed to the decision-support agent.

Every tool is a plain Python function returning JSON-serialisable data, so the
same functions back the deterministic orchestrator and the LLM tool-use loop.
Every call is recorded in `ToolBox.trace` for auditability.
"""
from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..data import load_dataset
from ..uncertainty import ClassConditionalConformal
from .knowledge import EvidenceRetriever

EPISTEMIC_HIGH = 0.08  # member-probability std above which the ensemble disagrees


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class ToolBox:
    def __init__(self, bundle_path=config.MODELS_DIR / "fusion_bundle.pkl"):
        with open(bundle_path, "rb") as f:
            b = pickle.load(f)
        self.model = b["model"]
        self.conformal: ClassConditionalConformal = b["conformal"]
        self.threshold = b["threshold_sens95"]
        self.train_ids = set(b["train_case_ids"])
        self.demo_ids = list(b["demo_case_ids"])
        self.df = load_dataset().set_index("case_id")
        self.retriever = EvidenceRetriever()
        self.trace: list[dict] = []
        self.retrieved_ids: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _row(self, case_id: str) -> pd.DataFrame:
        if case_id not in self.df.index:
            raise KeyError(f"Unknown case_id {case_id}")
        return self.df.loc[[case_id]]

    def call(self, name: str, **kwargs) -> dict:
        fn = getattr(self, name)
        try:
            result = fn(**kwargs)
        except Exception as e:  # tool errors are returned to the caller, not raised
            result = {"error": f"{type(e).__name__}: {e}"}
        self.trace.append({"tool": name, "input": kwargs, "output": result})
        return result

    # ------------------------------------------------------------------ tools
    def get_case(self, case_id: str) -> dict:
        row = self._row(case_id).iloc[0]
        modalities = {}
        for m, cols in config.MODALITIES.items():
            vals = {c: _clean(row[c]) for c in cols}
            modalities[m] = {"available": any(v is not None for v in vals.values()), "features": vals}
        return {
            "case_id": case_id,
            "center": row[config.CENTER],
            "psa_ng_ml": _clean(row["psa"]),
            "modalities": modalities,
            "missing_modalities": [m for m, v in modalities.items() if not v["available"]],
            "missing_features": [c for m in modalities.values() for c, v in m["features"].items() if v is None],
            "seen_during_training": case_id in self.train_ids,
        }

    def predict_risk(self, case_id: str, exclude_modalities: list[str] | None = None) -> dict:
        drop = tuple(exclude_modalities or [])
        pred = self.model.predict(self._row(case_id), drop=drop)
        p = float(pred.prob[0])
        sets = self.conformal.predict_sets(pred.prob)[0]
        std = float(pred.epistemic_std[0])
        return {
            "probability_cspca": round(p, 3),
            "epistemic_std": round(std, 3),
            "ensemble_disagreement_high": std > EPISTEMIC_HIGH,
            "conformal_set_90": self.conformal.describe(sets),
            "above_sensitivity95_threshold": p >= self.threshold,
            "sensitivity95_threshold": round(self.threshold, 3),
            "modality_attention": {m: round(float(a), 3) for m, a in zip(pred.modality_names, pred.attention[0])},
            "excluded_modalities": list(drop),
        }

    def modality_ablation(self, case_id: str) -> dict:
        """How much does the risk estimate change when each modality is withheld?"""
        base = self.predict_risk(case_id)["probability_cspca"]
        avail = [m for m, v in self.get_case(case_id)["modalities"].items() if v["available"]]
        out = {}
        for m in avail:
            if len(avail) == 1:
                break
            p = self.predict_risk(case_id, [m])["probability_cspca"]
            out[m] = {"probability_without": p, "change": round(p - base, 3)}
        return {"baseline_probability": base, "without_modality": out}

    def find_similar_cases(self, case_id: str, k: int = 15) -> dict:
        """Nearest training cases in the model's standardised feature space (case-based evidence)."""
        pool = self.df.loc[sorted(self.train_ids)]
        pool = pool[pool[config.GROUP] != self.df.loc[case_id, config.GROUP]]
        feats = self.model.prep.transform(pool)
        q = self.model.prep.transform(self._row(case_id))
        X = np.concatenate([feats[m] for m in feats], 1)
        xq = np.concatenate([q[m] for m in q], 1)
        d = np.linalg.norm(X - xq, axis=1)
        idx = np.argsort(d)[:k]
        nn = pool.iloc[idx]
        return {
            "k": k,
            "cspca_rate_among_similar": round(float(nn[config.TARGET].mean()), 3),
            "n_cspca": int(nn[config.TARGET].sum()),
            "biopsied_fraction": round(float(nn["biopsied"].mean()), 3),
            "examples": [
                {"case_id": cid, "pirads_max": _clean(r["pirads_max"]), "psad": _clean(r["psad_reported"]),
                 "cspca": int(r[config.TARGET])}
                for cid, r in nn.head(5).iterrows()
            ],
        }

    def retrieve_evidence(self, query: str, k: int = 4) -> dict:
        hits = self.retriever.search(query, k)
        self.retrieved_ids.update(h["id"] for h in hits)
        return {"query": query, "results": hits}


# JSON-schema definitions for the LLM tool-use loop
TOOL_SCHEMAS = [
    {
        "name": "get_case",
        "description": "Return the available clinical, radiology (PI-RADS) and MRI-derived imaging features "
                       "for a case, and which modalities or features are missing.",
        "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}},
                         "required": ["case_id"], "additionalProperties": False},
    },
    {
        "name": "predict_risk",
        "description": "Run the calibrated multimodal fusion ensemble. Returns probability of csPCa (ISUP>=2), "
                       "ensemble disagreement (epistemic uncertainty), the 90% class-conditional conformal "
                       "prediction set, and per-modality attention weights. Optionally exclude modalities.",
        "input_schema": {"type": "object", "properties": {
            "case_id": {"type": "string"},
            "exclude_modalities": {"type": "array", "items": {"type": "string",
                                                              "enum": list(config.MODALITIES)}}},
            "required": ["case_id"], "additionalProperties": False},
    },
    {
        "name": "modality_ablation",
        "description": "Recompute the risk with each available modality withheld, to show which modality drives "
                       "the estimate.",
        "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}},
                         "required": ["case_id"], "additionalProperties": False},
    },
    {
        "name": "find_similar_cases",
        "description": "Retrieve the most similar training cases and their observed csPCa outcomes.",
        "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"},
                                                          "k": {"type": "integer", "minimum": 5, "maximum": 50}},
                         "required": ["case_id"], "additionalProperties": False},
    },
    {
        "name": "retrieve_evidence",
        "description": "Search the curated guideline/literature evidence base. Returns snippets with ids that "
                       "must be cited as [ID] when used.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                          "k": {"type": "integer", "minimum": 1, "maximum": 6}},
                         "required": ["query"], "additionalProperties": False},
    },
]


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_clean)
