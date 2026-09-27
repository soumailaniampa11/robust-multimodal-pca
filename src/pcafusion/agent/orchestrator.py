"""Deterministic tool-using decision-support agent.

The plan is fixed and inspectable; branching depends on the tool outputs
(missing modalities, conformal set, ensemble disagreement):

    get_case -> predict_risk -> [modality_ablation] -> find_similar_cases
             -> retrieve_evidence (query built from the case) -> report

The agent never issues a clinical recommendation. It produces a *triage flag*
(low / elevated / uncertain) with the evidence behind it, and explicitly
escalates uncertain cases to human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .knowledge import EvidenceRetriever
from .tools import ToolBox

DISCLAIMER = (
    "> **Research demonstrator — not for clinical use.** Trained on the public PI-CAI dataset only, "
    "without prospective or external validation. Evidence snippets are paraphrases for demonstration."
)

FLAG_TEXT = {
    "low": "LOW estimated risk: the 90% conformal set contains only *no csPCa*.",
    "elevated": "ELEVATED estimated risk: the 90% conformal set contains only *csPCa*.",
    "uncertain": "UNCERTAIN: the evidence does not support a confident call; "
                 "escalate to multidisciplinary review.",
}


@dataclass
class AgentResult:
    case_id: str
    flag: str
    report_md: str
    trace: list[dict] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    citation_check: dict = field(default_factory=dict)


def triage_flag(risk: dict, missing_modalities: list[str] | None = None) -> tuple[str, list[str]]:
    """Map model outputs to a triage flag, with the reasons for any escalation.

    Safety rules: a LOW flag is only allowed when the conformal guarantee applies
    (it was calibrated on complete cases, so not when a modality is missing) and
    when it agrees with the high-sensitivity operating threshold.
    """
    reasons = []
    if risk["conformal_set_90"] in ("ambiguous", "atypical"):
        reasons.append(f"conformal prediction set is '{risk['conformal_set_90']}'")
    if risk["ensemble_disagreement_high"]:
        reasons.append("ensemble members disagree (high epistemic uncertainty)")
    if risk["conformal_set_90"] == "csPCa" and not reasons:
        return "elevated", reasons
    if risk["conformal_set_90"] == "no csPCa":
        if missing_modalities:
            reasons.append(f"missing modality ({', '.join(missing_modalities)}): the conformal guarantee was "
                           "calibrated on complete cases and does not apply")
        if risk["above_sensitivity95_threshold"]:
            reasons.append("probability is above the 95%-sensitivity threshold, conflicting with a negative set")
    return ("uncertain" if reasons else "low"), reasons


def evidence_query(case: dict) -> str:
    feats = case["modalities"]
    pirads = feats["radiology"]["features"]["pirads_max"]
    psad = feats["clinical"]["features"]["psad_reported"]
    terms = ["clinically significant prostate cancer biopsy decision"]
    if pirads is None:
        terms.append("MRI before biopsy PI-RADS assessment")
    elif pirads <= 2:
        terms.append("negative MRI PI-RADS 1-2 omit biopsy shared decision-making low suspicion")
    elif pirads == 3:
        terms.append("PI-RADS 3 equivocal PSA density biopsy")
    else:
        terms.append("positive MRI PI-RADS 4 5 targeted biopsy")
    if psad is not None:
        terms.append("PSA density threshold")
    if case["missing_modalities"]:
        terms.append("uncertainty limitations")
    return " ".join(terms)


def check_citations(text: str, retrieved: set[str], known: set[str]) -> dict:
    cited = sorted(set(re.findall(r"\[([A-Z]+-\d+)\]", text)))
    return {
        "cited": cited,
        "unknown_ids": [c for c in cited if c not in known],
        "not_retrieved_ids": [c for c in cited if c in known and c not in retrieved],
        "passed": all(c in retrieved for c in cited),
    }


def _fmt(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_report(case: dict, risk: dict, ablation: dict | None, similar: dict, evidence: dict, flag: str,
                  reasons: list[str] | None = None) -> str:
    f = case["modalities"]
    lines = [f"# Decision-support summary — case `{case['case_id']}`", "", DISCLAIMER, "",
             f"**Triage flag:** {FLAG_TEXT[flag]}"]
    for r in reasons or []:
        lines.append(f"- Escalation reason: {r}")
    lines += ["", "## 1. Data used"]
    lines.append("| Modality | Available | Features |")
    lines.append("|---|---|---|")
    for m, v in f.items():
        feats = ", ".join(f"{k}={_fmt(x)}" for k, x in v["features"].items() if k != "log_psa")
        if m == "clinical":
            feats = f"PSA={_fmt(case['psa_ng_ml'])} ng/mL, " + feats
        lines.append(f"| {m} | {'yes' if v['available'] else '**missing**'} | {feats} |")
    if case["missing_modalities"]:
        lines.append(f"\nMissing modalities ({', '.join(case['missing_modalities'])}) were **excluded, not imputed**; "
                     "the estimate relies on the remaining evidence only.")

    lines += ["", "## 2. Risk estimate and uncertainty",
              f"- Calibrated probability of csPCa (ISUP ≥ 2): **{risk['probability_cspca']:.2f}**",
              f"- 90% class-conditional conformal prediction set: **{risk['conformal_set_90']}**",
              f"- Ensemble disagreement (std of member probabilities): {risk['epistemic_std']:.3f}"
              + (" — **high**" if risk["ensemble_disagreement_high"] else ""),
              f"- Above the operating threshold targeting 95% sensitivity ({risk['sensitivity95_threshold']:.2f}): "
              f"{'yes' if risk['above_sensitivity95_threshold'] else 'no'}"]

    lines += ["", "## 3. What drives the estimate",
              "Modality attention weights: " + ", ".join(f"{m} {a:.2f}" for m, a in risk["modality_attention"].items())]
    if ablation and ablation["without_modality"]:
        lines.append("")
        lines.append("| Withheld modality | Probability | Change |")
        lines.append("|---|---|---|")
        for m, v in ablation["without_modality"].items():
            lines.append(f"| {m} | {v['probability_without']:.2f} | {v['change']:+.2f} |")

    lines += ["", "## 4. Similar cases (training cohort)",
              f"{similar['n_cspca']}/{similar['k']} most similar cases had csPCa "
              f"(rate {similar['cspca_rate_among_similar']:.2f}; {similar['biopsied_fraction']:.0%} biopsied)."]

    lines += ["", "## 5. Relevant evidence"]
    pirads = f["radiology"]["features"]["pirads_max"]
    ids = [h["id"] for h in evidence["results"]]
    for h in evidence["results"]:
        lines.append(f"- [{h['id']}] {h['text']} — *{h['source']}*")
    if pirads is not None:
        cat = "PIRADS-1" if "PIRADS-1" in ids else (ids[0] if ids else "")
        lines.append(f"\nThe radiologist's maximum PI-RADS category is {int(pirads)}"
                     + (f" [{cat}]." if cat else "."))

    lines += ["", "## 6. Limitations",
              "- Single public dataset (3 Dutch centres), retrospective; PI-RADS ≤ 2 negatives were not biopsied "
              "(verification bias).",
              "- No PSMA PET/CT, histopathology images or longitudinal EHR in this dataset.",
              "- Imaging features are volumes from AI segmentations, not learned image representations.",
              "- The flag supports, and does not replace, clinical judgement and shared decision-making [AI-1]."
              if "AI-1" in ids else
              "- The flag supports, and does not replace, clinical judgement and shared decision-making."]
    return "\n".join(lines)


class DecisionSupportAgent:
    def __init__(self, toolbox: ToolBox | None = None):
        self.tools = toolbox or ToolBox()

    def run(self, case_id: str) -> AgentResult:
        t = self.tools
        t.trace, t.retrieved_ids = [], set()
        case = t.call("get_case", case_id=case_id)
        if "error" in case:
            raise KeyError(case["error"])
        risk = t.call("predict_risk", case_id=case_id)
        flag, reasons = triage_flag(risk, case["missing_modalities"])
        n_avail = sum(v["available"] for v in case["modalities"].values())
        # Explain the drivers whenever the case is uncertain or more than one modality is present.
        ablation = t.call("modality_ablation", case_id=case_id) if n_avail > 1 else None
        similar = t.call("find_similar_cases", case_id=case_id)
        evidence = t.call("retrieve_evidence", query=evidence_query(case), k=5)
        if flag == "uncertain":
            extra = t.call("retrieve_evidence", query="uncertain AI output escalate multidisciplinary review", k=1)
            seen = {h["id"] for h in evidence["results"]}
            evidence["results"] += [h for h in extra["results"] if h["id"] not in seen]
        report = render_report(case, risk, ablation, similar, evidence, flag, reasons)
        check = check_citations(report, t.retrieved_ids, set(EvidenceRetriever().by_id))
        return AgentResult(case_id, flag, report, list(t.trace), check["cited"], check)
