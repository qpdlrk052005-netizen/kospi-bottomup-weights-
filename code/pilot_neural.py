"""Neural models for the daily KOSPI pilot, revision 2."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pilot_data import LOOKBACK, MARKET_FEATURES
from pilot_fit import RegressionHead

HIDDEN = 64
HEADS = 4
PATCH = 5
AUX_WEIGHTS = (0.0, 0.1)
MARKET_ENCODERS = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
GRAPH_SPEC = "corr_knn_change_60d_min40_k10_undirected_self_v2"


@dataclass(frozen=True)
class Setting:
    cross_firm: str
    market: str
    aux_weight: float

    @property
    def structure(self):
        return f"{self.cross_firm}_{self.market}"

    @property
    def name(self):
        return f"{self.structure}_aux{self.aux_weight:g}"


SETTINGS = tuple(
    Setting(cross_firm, encoder, aux)
    for cross_firm in ("A", "A-i", "B")
    for encoder in MARKET_ENCODERS
    for aux in AUX_WEIGHTS
)


class CorrelationGraphs:
    """Precomputed yearly shards of the strictly trailing correlation kNN graph."""

    def __init__(self, data, root):
        self.data = data
        self.root = Path(root)
        self.loaded = {}
        self.change = data.stock[:, :, data.features.index("Change")]

    @staticmethod
    def _identity(data):
        locked = data.locked_inputs()
        identity = {"file_sha256": locked["file_sha256"], "graph_spec": GRAPH_SPEC}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return digest, identity

    @classmethod
    def prepare(cls, data, output, log=None):
        digest, identity = cls._identity(data)
        root = Path(output) / "correlation_graphs" / digest[:16]
        root.mkdir(parents=True, exist_ok=True)
        if log:
            log(f"Preparing G_t cache at {root}")

        years = np.unique(data.years[data.labeled])
        shard_files = []
        for year in years:
            days = np.flatnonzero(data.labeled & (data.years == year)).astype(np.int32)
            path = root / f"graphs_{int(year)}.npz"
            reusable = False
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        reusable = np.array_equal(archive["days"], days)
                except (OSError, ValueError, KeyError):
                    reusable = False
            if not reusable:
                edge_chunks = []
                offsets = [0]
                builder = cls(data, root)
                for j, day in enumerate(days, start=1):
                    edge = builder._build(int(day))
                    edge_chunks.append(edge)
                    offsets.append(offsets[-1] + edge.shape[1])
                    if log and j % 50 == 0:
                        log(f"G_t year={year}: {j}/{len(days)} dates")
                edges = np.concatenate(edge_chunks, axis=1) if edge_chunks else np.empty((2, 0), dtype=np.int32)
                temporary = path.with_name(path.stem + ".temporary.npz")
                np.savez_compressed(
                    temporary,
                    days=days,
                    offsets=np.asarray(offsets, dtype=np.int64),
                    edges=edges.astype(np.int32, copy=False),
                )
                temporary.replace(path)
                if log:
                    log(f"G_t year={year}: saved {len(days)} dates, {edges.shape[1]} edges")
            shard_files.append(path.name)

        manifest = {**identity, "shards": shard_files, "date_count": int(data.labeled.sum())}
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != manifest:
                raise ValueError("The graph cache identity changed while it was being prepared.")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return cls(data, root)

    def _build(self, day):
        ids = self.data.members[day]
        if len(ids) == 0:
            return np.empty((2, 0), dtype=np.int32)
        days = np.arange(day - LOOKBACK + 1, day + 1)
        valid = self.data.valid[days[:, None], ids].T.astype(bool)
        daily_change = self.change[days[:, None], ids].T
        valid &= np.isfinite(daily_change)
        x = np.where(valid, daily_change, 0.0).astype(np.float64)
        valid_float = valid.astype(np.float64)
        count = valid_float @ valid_float.T
        safe_count = np.maximum(1.0, count)
        first = x @ valid_float.T
        second = (x * x) @ valid_float.T
        numerator = x @ x.T - first * first.T / safe_count
        variance = np.maximum(0.0, second - first * first / safe_count)
        denominator = np.sqrt(variance * variance.T)
        admissible = (count >= 40) & (denominator > 1e-18)
        correlation = np.full(count.shape, -np.inf, dtype=np.float64)
        np.divide(numerator, denominator, out=correlation, where=admissible)
        np.fill_diagonal(correlation, -np.inf)

        k = min(10, max(0, len(ids) - 1))
        linked = np.eye(len(ids), dtype=bool)
        if k:
            for row in range(len(ids)):
                candidates = np.flatnonzero(np.isfinite(correlation[row]))
                if len(candidates) <= k:
                    chosen = candidates
                else:
                    values = correlation[row, candidates]
                    cutoff = np.partition(values, len(values) - k)[len(values) - k]
                    above = candidates[values > cutoff]
                    tied = candidates[values == cutoff]
                    chosen = np.concatenate((above, tied[:k - len(above)]))
                linked[row, chosen] = True
        linked |= linked.T
        source, target = np.where(linked)
        return np.stack((source, target)).astype(np.int32)

    def __getitem__(self, day):
        year = int(self.data.years[day])
        if year not in self.loaded:
            path = self.root / f"graphs_{year}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing precomputed G_t shard: {path}")
            with np.load(path, allow_pickle=False) as archive:
                days = archive["days"]
                offsets = archive["offsets"]
                edges = archive["edges"]
            day_to_row = {int(value): i for i, value in enumerate(days)}
            self.loaded[year] = (day_to_row, offsets, edges)
        day_to_row, offsets, edges = self.loaded[year]
        row = day_to_row.get(int(day))
        if row is None:
            raise KeyError(f"No precomputed G_t for date {self.data.dates[day]}.")
        start, end = offsets[row:row + 2]
        return edges[:, start:end]


class NeuralBatches:
    def __init__(self, data, folded, device, graphs=None):
        self.data, self.folded, self.device, self.graphs = data, folded, device, graphs

    def __call__(self, days):
        days = np.asarray(days, dtype=np.int64)
        dates = days[:, None] + np.arange(-LOOKBACK + 1, 1)
        ids = [self.data.members[d] for d in days]
        lengths = list(map(len, ids))
        if not lengths or min(lengths) < 1:
            raise ValueError("Every training/evaluation date must have a nonempty U_t.")
        width = max(lengths)
        p = self.folded.stock.shape[-1]
        x_current = np.zeros((len(days), width, p), dtype=np.float32)
        log_cap = np.full((len(days), width), -np.inf, dtype=np.float32)
        cap_pct = np.zeros((len(days), width), dtype=np.float32)
        aux = np.zeros((len(days), width), dtype=np.float32)
        aux_valid = np.zeros((len(days), width), dtype=bool)
        temporal, observed, edge_list = [], [], []
        node_offset = 0

        for i, (day, window, codes, n) in enumerate(zip(days, dates, ids, lengths, strict=True)):
            in_universe = self.data.universe[window[:, None], codes[None, :]].T
            hist = self.folded.stock[window[None, :], codes[:, None]].copy()
            hist[~in_universe] = 0.0
            rank = self.folded.cap_pct[window[None, :], codes[:, None]]
            rank = np.where(in_universe, rank, 0.0).astype(np.float32)
            indicator = in_universe.astype(np.float32)
            temporal.append(np.concatenate((hist, rank[..., None], indicator[..., None]), axis=-1))
            # T-patch masks only dates with Valid=False. U_t membership remains
            # the separate indicator feature and gates x/CapPct zero filling.
            observed.append(self.data.valid[window[:, None], codes[None, :]].T)

            current_x = self.folded.stock[day, codes]
            x_current[i, :n] = current_x
            cap_pct[i, :n] = self.folded.cap_pct[day, codes]
            cap = self.data.cap[day, codes]
            cap_weight = cap / cap.sum()
            log_cap[i, :n] = np.log(cap_weight).astype(np.float32)

            targets = self.data.firm_y[day, codes]
            aux_valid[i, :n] = np.isfinite(targets)
            aux[i, :n] = np.nan_to_num(targets, nan=0.0)
            if self.graphs is not None:
                edge_list.append(self.graphs[int(day)] + node_offset)
            node_offset += n

        packed = {
            "x": x_current,
            "history": np.concatenate(temporal, axis=0),
            "observed": np.concatenate(observed, axis=0),
            "valid": np.arange(width)[None, :] < np.asarray(lengths)[:, None],
            "log_cap": log_cap,
            "cap_pct": cap_pct,
            "market_history": self.folded.market[dates],
            "s": self.folded.market[days],
            "y": self.data.y[days].astype(np.float32),
            "aux": aux,
            "aux_valid": aux_valid,
        }
        if edge_list:
            packed["edges"] = np.concatenate(edge_list, axis=1).astype(np.int64)
        return {
            **{name: torch.as_tensor(value, device=self.device) for name, value in packed.items()},
            "lengths": lengths,
        }


def transformer_layer():
    return nn.TransformerEncoderLayer(
        HIDDEN,
        HEADS,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
    )


class PatchTemporalEncoder(nn.Module):
    def __init__(self, input_count):
        super().__init__()
        self.projection = nn.Linear(PATCH * input_count, HIDDEN)
        self.position = nn.Parameter(torch.empty(1, LOOKBACK // PATCH, HIDDEN))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList([transformer_layer() for _ in range(2)])

    def forward(self, history, observed):
        n = history.shape[0]
        patch_valid = observed.reshape(n, LOOKBACK // PATCH, PATCH).any(-1)
        x = history.masked_fill(~observed[..., None], 0.0)
        embedded = self.projection(x.reshape(n, LOOKBACK // PATCH, -1)) + self.position
        for layer in self.layers:
            embedded = layer(embedded, src_key_padding_mask=~patch_valid)
        return (embedded * patch_valid[..., None]).sum(1) / patch_valid.sum(1, keepdim=True)


class LinearTemporalEncoder(nn.Module):
    def __init__(self, input_count):
        super().__init__()
        self.projection = nn.Linear(LOOKBACK * input_count, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, history, observed):
        return self.norm(self.projection(history.reshape(history.shape[0], -1)))


class GraphAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.out = nn.Linear(HIDDEN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)
        self.in_attention = nn.Parameter(torch.empty(HEADS, HIDDEN // HEADS))
        self.out_attention = nn.Parameter(torch.empty_like(self.in_attention))
        nn.init.xavier_uniform_(self.in_attention)
        nn.init.xavier_uniform_(self.out_attention)

    def forward(self, features, edges):
        source, target = edges
        n = features.shape[0]
        projected = self.project(F.dropout(features, p=0.1, training=self.training)).reshape(n, HEADS, -1)
        source_score = (projected * self.out_attention).sum(dim=-1)
        target_score = (projected * self.in_attention).sum(dim=-1)
        logits = F.leaky_relu(source_score[source] + target_score[target], negative_slope=0.2)
        target_heads = target[:, None].expand(-1, HEADS)
        with torch.no_grad():
            maximum = logits.new_full((n, HEADS), -torch.inf)
            maximum.scatter_reduce_(0, target_heads, logits, reduce="amax", include_self=True)
        numerator = (logits - maximum[target]).exp()
        denominator = logits.new_zeros((n, HEADS)).index_add_(0, target, numerator)
        attention = F.dropout(numerator / denominator[target], p=0.1, training=self.training)
        contributions = projected[source] * attention[..., None]
        aggregate = torch.zeros_like(projected).index_add_(0, target, contributions)
        return self.norm(features + self.out(F.elu(aggregate.reshape(n, HIDDEN))))


class DynamicModel(nn.Module):
    def __init__(self, setting, firm_count, alpha_init):
        super().__init__()
        self.setting = setting
        input_count = firm_count + 2
        self.temporal = (
            LinearTemporalEncoder(input_count)
            if setting.cross_firm == "A-i"
            else PatchTemporalEncoder(input_count)
        )
        if setting.cross_firm in ("A", "A-i"):
            self.cross_firm = transformer_layer()
        else:
            self.cross_firm = nn.ModuleList([GraphAttentionLayer(), GraphAttentionLayer()])
        self.market = MARKET_ENCODERS[setting.market](len(MARKET_FEATURES), 32, num_layers=1, batch_first=True)
        self.firm_score = nn.Linear(HIDDEN, 32)
        self.market_score = nn.Linear(32, 32, bias=False)
        self.cap_score = nn.Linear(1, 32, bias=False)
        self.weight_score = nn.Linear(32, 1, bias=False)
        nn.init.zeros_(self.weight_score.weight)
        self.head = RegressionHead(firm_count, len(MARKET_FEATURES), alpha_init, zero_init=True)
        self.aux_head = nn.Linear(HIDDEN, 1) if setting.aux_weight else None

    def forward(self, batch):
        valid = batch["valid"]
        embedding = self.temporal(batch["history"], batch["observed"])
        if self.setting.cross_firm == "B":
            for layer in self.cross_firm:
                embedding = layer(embedding, batch["edges"])

        firms = embedding.new_zeros((*valid.shape, HIDDEN))
        firms[valid] = embedding
        if self.setting.cross_firm in ("A", "A-i"):
            firms = self.cross_firm(firms, src_key_padding_mask=~valid)

        states, _ = self.market(batch["market_history"])
        score = torch.tanh(
            self.firm_score(firms)
            + self.market_score(states[:, -1])[:, None, :]
            + self.cap_score(batch["cap_pct"][..., None])
        )
        delta = self.weight_score(score).squeeze(-1)
        weight = self.weights(batch, delta)
        forecast, variance = self.head(batch["x"], batch["s"], weight)
        result = {"prediction": forecast, "variance": variance, "weights": weight, "delta": delta}
        if self.aux_head is not None:
            result["aux_prediction"] = self.aux_head(firms).squeeze(-1)
        return result

    @staticmethod
    def weights(batch, delta):
        logits = (batch["log_cap"] + delta).masked_fill(~batch["valid"], -torch.inf)
        return torch.softmax(logits, dim=-1)

    def objective(self, out, batch):
        loss = ((batch["y"] - out["prediction"]).square() + out["variance"]).mean()
        if self.aux_head is None:
            return loss
        squared_error = (batch["aux"] - out["aux_prediction"]).square()
        present = batch["aux_valid"].sum(dim=-1)
        present_days = (present > 0).sum().clamp_min(1)
        sum_by_day = (squared_error * batch["aux_valid"]).sum(dim=-1)
        aux_loss = (sum_by_day / present.clamp_min(1)).sum() / present_days
        return loss + self.setting.aux_weight * aux_loss

    def ablate(self, output, batch):
        changed = output["delta"].clone()
        for i, n in enumerate(batch["lengths"]):
            k = max(1, (n + 9) // 10)
            top = torch.argsort(changed[i, :n], descending=True, stable=True)[:k]
            changed[i, top] = 0.0
        forecast, _ = self.head(batch["x"], batch["s"], self.weights(batch, changed))
        return forecast
