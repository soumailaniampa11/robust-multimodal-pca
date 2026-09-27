"""Small curated evidence base used for retrieval-grounded explanations.

Each entry is a short *paraphrase* of a guideline statement or key study,
written for this demonstrator. They are not verbatim quotes and must be checked
against the current source documents before any real-world use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    text: str


EVIDENCE: list[Evidence] = [
    Evidence("PIRADS-1", "PI-RADS v2.1 (Turkbey et al., Eur Urol 2019)",
             "PI-RADS assessment categories express the likelihood of clinically significant cancer: "
             "1 very low, 2 low, 3 intermediate (equivocal), 4 high (likely), 5 very high (highly likely)."),
    Evidence("PIRADS-2", "PI-RADS v2.1 (Turkbey et al., Eur Urol 2019)",
             "PI-RADS 3 lesions are equivocal; the decision to biopsy is commonly informed by additional "
             "factors such as PSA density, prior biopsy history and patient preference."),
    Evidence("EAU-1", "EAU-EANM-ESTRO-ESUR-ISUP-SIOG Guidelines on Prostate Cancer",
             "Multiparametric or biparametric MRI is recommended before prostate biopsy in biopsy-naive men "
             "with suspected prostate cancer."),
    Evidence("EAU-2", "EAU-EANM-ESTRO-ESUR-ISUP-SIOG Guidelines on Prostate Cancer",
             "When MRI is positive (PI-RADS 3 or higher), MRI-targeted biopsy is recommended, "
             "combined with systematic biopsy in biopsy-naive men."),
    Evidence("EAU-3", "EAU-EANM-ESTRO-ESUR-ISUP-SIOG Guidelines on Prostate Cancer",
             "When MRI is negative (PI-RADS 1-2) and clinical suspicion of prostate cancer is low, "
             "biopsy may be omitted based on shared decision-making with the patient."),
    Evidence("EAU-4", "EAU-EANM-ESTRO-ESUR-ISUP-SIOG Guidelines on Prostate Cancer",
             "Risk calculators and PSA density can be used to stratify men and help avoid unnecessary biopsies "
             "and overdiagnosis of insignificant cancer."),
    Evidence("PSAD-1", "Schoots et al., BJU Int 2021 (risk-adapted biopsy decisions)",
             "PSA density thresholds around 0.15-0.20 ng/mL/cc are associated with a higher probability of "
             "clinically significant cancer; low PSA density with negative or equivocal MRI identifies men at "
             "low risk who may safely avoid immediate biopsy."),
    Evidence("PRECISION-1", "Kasivisvanathan et al., NEJM 2018 (PRECISION trial)",
             "An MRI-first pathway with targeted biopsy detected more clinically significant and fewer "
             "clinically insignificant cancers than standard systematic TRUS biopsy, while allowing some men "
             "to avoid biopsy."),
    Evidence("PICAI-1", "Saha et al., Lancet Oncol 2024 (PI-CAI)",
             "In a retrospective international study, an AI system for prostate MRI was non-inferior to "
             "radiologists at detecting clinically significant cancer; prospective validation is required "
             "before clinical use."),
    Evidence("DEF-1", "ISUP 2014 grading / PI-CAI reference standard",
             "Clinically significant prostate cancer (csPCa) is defined as ISUP grade group 2 or higher."),
    Evidence("VOL-1", "Prostate MRI literature (gland volume and PSA density)",
             "Larger prostate volume, often due to benign transition-zone hyperplasia, raises PSA "
             "independently of cancer; normalising PSA by gland volume (PSA density) corrects for this."),
    Evidence("AI-1", "Principles for AI-based clinical decision support",
             "AI outputs should be presented with calibrated uncertainty, the evidence used, and known "
             "limitations; the treating clinician remains responsible for the decision, and uncertain cases "
             "should be escalated to multidisciplinary review."),
]


class EvidenceRetriever:
    """TF-IDF retrieval over the curated evidence base."""

    def __init__(self, evidence: list[Evidence] = EVIDENCE):
        self.evidence = evidence
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([f"{e.source} {e.text}" for e in evidence])
        self.by_id = {e.id: e for e in evidence}

    def search(self, query: str, k: int = 4) -> list[dict]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [
            {"id": self.evidence[i].id, "source": self.evidence[i].source, "text": self.evidence[i].text,
             "score": round(float(scores[i]), 3)}
            for i in order if scores[i] > 0
        ]
