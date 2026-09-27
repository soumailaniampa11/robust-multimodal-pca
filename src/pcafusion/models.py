"""Missing-modality-aware multimodal fusion model and baselines.

Architecture (per case):
    modality m  ->  [standardised features ‖ per-feature observed mask]  ->  MLP encoder  ->  e_m
    attention score a_m = v^T tanh(W e_m), softmax over *available* modalities only
    fused = sum_m alpha_m e_m  ->  classifier  ->  logit

Design choices:
* A modality that is absent is excluded from the attention softmax instead of
  being imputed, so the prediction is based only on observed evidence and the
  attention weights alpha_m report how much each available modality contributed.
* Modality dropout during training teaches the model to predict from any
  subset of modalities.
* A deep ensemble (independently initialised members) gives an epistemic
  uncertainty estimate; temperature scaling calibrates the averaged logit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config
from .uncertainty import TemperatureScaler

LOG_FEATURES = {
    "psad_reported", "volume_reported", "gland_volume_seg", "pz_volume_seg", "tz_volume_seg",
}


class ModalityPreprocessor:
    """Per-modality standardisation that keeps track of missing values."""

    def __init__(self, modalities: dict[str, list[str]] = config.MODALITIES):
        self.modalities = modalities
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    @staticmethod
    def _raw(df: pd.DataFrame, col: str) -> np.ndarray:
        x = df[col].to_numpy(dtype=float)
        return np.log1p(x) if col in LOG_FEATURES else x

    def fit(self, df: pd.DataFrame) -> "ModalityPreprocessor":
        for cols in self.modalities.values():
            for col in cols:
                x = self._raw(df, col)
                self.mean_[col] = float(np.nanmean(x))
                self.std_[col] = float(np.nanstd(x) + 1e-6)
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return {modality: array (n, 2 * n_features)} = [values ‖ observed mask]."""
        out = {}
        for m, cols in self.modalities.items():
            vals, masks = [], []
            for col in cols:
                x = (self._raw(df, col) - self.mean_[col]) / self.std_[col]
                observed = ~np.isnan(x)
                vals.append(np.where(observed, x, 0.0))
                masks.append(observed.astype(float))
            out[m] = np.concatenate([np.stack(vals, 1), np.stack(masks, 1)], axis=1).astype(np.float32)
        return out

    def availability(self, df: pd.DataFrame) -> np.ndarray:
        """(n, n_modalities) bool: modality has at least one observed feature."""
        return np.stack(
            [df[cols].notna().any(axis=1).to_numpy() for cols in self.modalities.values()], axis=1
        )


class MultimodalFusionNet(nn.Module):
    def __init__(self, modality_dims: dict[str, int], hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.names = list(modality_dims)
        self.encoders = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.GELU(),
                )
                for m, d in modality_dims.items()
            }
        )
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, inputs: dict[str, torch.Tensor], available: torch.Tensor):
        emb = torch.stack([self.encoders[m](inputs[m]) for m in self.names], dim=1)  # (n, M, h)
        scores = self.attn(emb).squeeze(-1).masked_fill(~available, float("-inf"))
        alpha = torch.softmax(scores, dim=1)  # (n, M); zero weight on missing modalities
        fused = (alpha.unsqueeze(-1) * emb).sum(1)
        return self.head(fused).squeeze(-1), alpha


@dataclass
class FusionPrediction:
    prob: np.ndarray            # calibrated ensemble probability
    epistemic_std: np.ndarray   # std of member probabilities
    attention: np.ndarray       # (n, M) mean modality attention weights
    modality_names: list[str] = field(default_factory=list)


class FusionEnsemble:
    """Deep ensemble of MultimodalFusionNet with temperature calibration."""

    def __init__(self, n_members: int = 5, hidden: int = 32, epochs: int = 200, lr: float = 1e-3,
                 weight_decay: float = 1e-4, modality_dropout: float = 0.3, patience: int = 25,
                 seed: int = config.SEED):
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.modality_dropout = modality_dropout
        self.patience = patience
        self.seed = seed
        self.prep = ModalityPreprocessor()
        self.members: list[MultimodalFusionNet] = []
        self.temperature = TemperatureScaler()

    # -- helpers -----------------------------------------------------------
    def _tensors(self, df: pd.DataFrame, drop: tuple[str, ...] = ()):
        inputs = {m: torch.from_numpy(x) for m, x in self.prep.transform(df).items()}
        avail = self.prep.availability(df)
        for m in drop:
            avail[:, self.prep_names.index(m)] = False
        return inputs, torch.from_numpy(avail)

    @property
    def prep_names(self) -> list[str]:
        return list(self.prep.modalities)

    def _drop_modalities(self, avail: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        drop = torch.rand(avail.shape, generator=gen) < self.modality_dropout
        new = avail & ~drop
        # keep at least one available modality per sample
        empty = new.sum(1) == 0
        new[empty] = avail[empty]
        return new

    def _train_member(self, k: int, tr, va, y_tr, y_va) -> MultimodalFusionNet:
        torch.manual_seed(self.seed + k)
        gen = torch.Generator().manual_seed(self.seed + k)
        dims = {m: x.shape[1] for m, x in tr[0].items()}
        net = MultimodalFusionNet(dims, hidden=self.hidden)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(y_tr)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                batch = {m: x[idx] for m, x in tr[0].items()}
                avail = self._drop_modalities(tr[1][idx], gen)
                logit, _ = net(batch, avail)
                loss = loss_fn(logit, y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(*va)[0], y_va).item()
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k_: v.clone() for k_, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    # -- public API --------------------------------------------------------
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> "FusionEnsemble":
        """Train on df_train; df_val is used for early stopping and temperature scaling."""
        self.prep.fit(df_train)
        tr, va = self._tensors(df_train), self._tensors(df_val)
        y_tr = torch.tensor(df_train[config.TARGET].to_numpy(), dtype=torch.float32)
        y_va = torch.tensor(df_val[config.TARGET].to_numpy(), dtype=torch.float32)
        self.members = [self._train_member(k, tr, va, y_tr, y_va) for k in range(self.n_members)]
        logits, *_ = self._member_outputs(df_val)
        self.temperature.fit(logits.mean(0), df_val[config.TARGET].to_numpy())
        return self

    def _member_outputs(self, df: pd.DataFrame, drop: tuple[str, ...] = ()):
        inputs, avail = self._tensors(df, drop)
        logits, alphas = [], []
        with torch.no_grad():
            for net in self.members:
                lg, a = net(inputs, avail)
                logits.append(lg.numpy())
                alphas.append(a.numpy())
        return np.stack(logits), np.stack(alphas)

    def predict(self, df: pd.DataFrame, drop: tuple[str, ...] = ()) -> FusionPrediction:
        """Predict, optionally simulating that modalities in `drop` are unavailable."""
        logits, alphas = self._member_outputs(df, drop)
        member_probs = 1.0 / (1.0 + np.exp(-logits))
        return FusionPrediction(
            prob=self.temperature.transform(logits.mean(0)),
            epistemic_std=member_probs.std(0),
            attention=alphas.mean(0),
            modality_names=self.prep_names,
        )

    def predict_proba(self, df: pd.DataFrame, drop: tuple[str, ...] = ()) -> np.ndarray:
        return self.predict(df, drop).prob
