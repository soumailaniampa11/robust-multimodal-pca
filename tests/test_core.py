"""Unit tests for the core components (run: pytest -q)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config  # noqa: E402
from pcafusion.agent.knowledge import EvidenceRetriever  # noqa: E402
from pcafusion.agent.orchestrator import check_citations, triage_flag  # noqa: E402
from pcafusion.data import parse_pirads  # noqa: E402
from pcafusion.evaluation import net_benefit, threshold_for_sensitivity  # noqa: E402
from pcafusion.models import FusionEnsemble  # noqa: E402
from pcafusion.uncertainty import ClassConditionalConformal, TemperatureScaler, expected_calibration_error  # noqa: E402


def test_parse_pirads():
    assert parse_pirads("4,2") == (4.0, 1)
    assert parse_pirads("5,N/A") == (5.0, 1)
    assert parse_pirads("2") == (2.0, 0)
    assert np.isnan(parse_pirads(np.nan)[0])


def test_conformal_class_conditional_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1)
    cal, test = slice(0, 2000), slice(2000, None)
    cp = ClassConditionalConformal(alpha=0.1).fit(p[cal], y[cal])
    sets = cp.predict_sets(p[test])
    yt = y[test]
    assert sets[yt == 1, 1].mean() >= 0.87
    assert sets[yt == 0, 0].mean() >= 0.87


def test_temperature_scaling_improves_overconfident_logits():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    logits = (2 * y - 1) * 1.0 + rng.normal(0, 1.5, 3000)
    over = logits * 4  # overconfident
    ts = TemperatureScaler().fit(over, y)
    assert ts.T > 2
    before = expected_calibration_error(y, 1 / (1 + np.exp(-over)))
    after = expected_calibration_error(y, ts.transform(over))
    assert after < before


def test_threshold_and_net_benefit():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    assert threshold_for_sensitivity(y, p, 1.0) == 0.3
    nb = net_benefit(y, p, np.array([0.25]))
    assert nb[0] == pytest.approx(4 / 6)


def _toy_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({config.TARGET: y})
    for m, cols in config.MODALITIES.items():
        for c in cols:
            df[c] = rng.normal(0, 1, n) + (1.5 * y if m == "radiology" else 0.3 * y)
    for c in ("volume_reported", "psad_reported", "gland_volume_seg", "pz_volume_seg", "tz_volume_seg"):
        df[c] = np.abs(df[c]) + 1
    df.loc[: n // 5, config.MODALITIES["imaging"]] = np.nan  # whole modality missing for 20%
    return df


def test_fusion_handles_missing_modalities():
    df = _toy_frame()
    model = FusionEnsemble(n_members=2, epochs=15).fit(df.iloc[:200], df.iloc[200:])
    pred = model.predict(df.iloc[:50])
    assert np.all((pred.prob > 0) & (pred.prob < 1))
    imaging = pred.modality_names.index("imaging")
    # the first 61 rows have no imaging: they must receive zero attention on it
    assert np.allclose(pred.attention[:, imaging], 0.0)
    dropped = model.predict(df.iloc[:50], drop=("radiology", "imaging"))
    assert np.allclose(dropped.attention[:, pred.modality_names.index("clinical")], 1.0)


def test_triage_safety_rules():
    base = {"conformal_set_90": "no csPCa", "ensemble_disagreement_high": False,
            "above_sensitivity95_threshold": False}
    assert triage_flag(base)[0] == "low"
    assert triage_flag(base, ["radiology"])[0] == "uncertain"
    assert triage_flag({**base, "above_sensitivity95_threshold": True})[0] == "uncertain"
    assert triage_flag({**base, "conformal_set_90": "ambiguous"})[0] == "uncertain"
    assert triage_flag({**base, "conformal_set_90": "csPCa", "above_sensitivity95_threshold": True})[0] == "elevated"


def test_citation_check_and_retrieval():
    r = EvidenceRetriever()
    hits = r.search("PI-RADS 3 equivocal PSA density", k=3)
    ids = {h["id"] for h in hits}
    assert "PIRADS-2" in ids
    ok = check_citations("see [PIRADS-2]", ids, set(r.by_id))
    bad = check_citations("see [EAU-9] and [PICAI-1]", ids, set(r.by_id))
    assert ok["passed"] and not bad["passed"]
    assert bad["unknown_ids"] == ["EAU-9"]
