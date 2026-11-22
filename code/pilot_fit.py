"""Common float32 loss and training procedure for the daily KOSPI pilot."""

from copy import deepcopy
import random
import time

import numpy as np
import torch
from torch import nn

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 16
MAX_EPOCHS = 20
PATIENCE = 3
SELECTION_SEEDS = (0, 1)
TEST_SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class RegressionHead(nn.Module):
    def __init__(self, firm_count, market_count, alpha_init=0.0, zero_init=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.beta = nn.Linear(firm_count, 1, bias=False)
        self.gamma = nn.Linear(market_count, 1, bias=False)
        if zero_init:
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.gamma.weight)

    def forward(self, x, s, weights):
        partial = self.beta(x).squeeze(-1)
        forecast = self.alpha + self.gamma(s).squeeze(-1) + (weights * partial).sum(-1)
        detached = weights.detach()
        centered = partial - (detached * partial).sum(-1, keepdim=True)
        variance = (detached * centered.square()).sum(-1)
        return forecast, variance

    def coefficients(self):
        return (
            self.alpha.detach().item(),
            self.beta.weight.detach().cpu().numpy()[0].copy(),
            self.gamma.weight.detach().cpu().numpy()[0].copy(),
        )


def fit(
    model,
    batches,
    train,
    validation,
    seed,
    epochs=MAX_EPOCHS,
    batch_size=BATCH,
    verbose=False,
    epoch_callback=None,
    early_stopping=True,
    restore_best=True,
):
    no_decay = [model.head.alpha]
    exempt = {id(param) for param in no_decay}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in exempt], "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    rng = np.random.default_rng(seed)
    history, best_epoch, best_state, best_value = [], None, None, np.inf
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        started = time.monotonic()
        model.train()
        order = rng.permutation(train)
        train_loss = 0.0
        for i in range(0, len(order), batch_size):
            days = order[i:i + batch_size]
            batch = batches(days)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = model.objective(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.detach().item() * len(days)

        mse = np.nan
        if validation is not None:
            forecast = evaluate(model, batches, validation, batch_size)["prediction"]
            mse = float(np.square(forecast - batches.data.y[validation]).mean())
            if np.isfinite(mse) and mse < best_value:
                best_value, best_epoch, best_state = mse, epoch, deepcopy(model.state_dict())

        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "val_mse": mse,
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(history, best_epoch)
        if verbose:
            print(
                f"  epoch={epoch:02d} train_loss={row['train_loss']:.6f} "
                f"val_mse={mse:.6f} seconds={row['seconds']:.2f}",
                flush=True,
            )
        if (
            validation is not None
            and early_stopping
            and best_epoch is not None
            and epoch - best_epoch >= PATIENCE
        ):
            break

    if validation is not None:
        if best_state is None:
            raise ValueError("All validation predictions were non-finite.")
        if restore_best:
            model.load_state_dict(best_state)
    return history, best_epoch, best_value


@torch.inference_mode()
def evaluate(model, batches, days, batch_size=BATCH, details=False):
    model.eval()
    forecasts, weights, deltas, ablated = [], [], [], []
    for i in range(0, len(days), batch_size):
        group = days[i:i + batch_size]
        batch = batches(group)
        out = model(batch)
        forecasts.extend(out["prediction"].cpu().numpy().tolist())
        if details:
            spans = batch["lengths"]
            for w, n in zip(out["weights"].cpu().numpy(), spans, strict=True):
                weights.append(w[:n].copy())
            if "delta" in out:
                for delta, n in zip(out["delta"].cpu().numpy(), spans, strict=True):
                    deltas.append(delta[:n].copy())
                ablated.extend(model.ablate(out, batch).cpu().numpy().tolist())
    return {
        "prediction": np.asarray(forecasts),
        "weights": weights,
        "deltas": deltas,
        "ablated": np.asarray(ablated),
    }
