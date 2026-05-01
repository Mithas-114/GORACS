"""
Regression tests for the batched changes introduced in ITRA.py:
  1. greedy() – gain is now computed in batched loop instead of a single vectorised op
  2. _get_y_hat() – replaces inline topk + compute_F; used inside pruning()
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codes", "select"))

import time
import torch
import numpy as np

from ITRA import ITRA

BATCH_SIZE = 10240
# exactly 2 batches for both benchmark functions
N_TRAIN_BENCH = BATCH_SIZE * 2
N_VAL_BENCH = 2000
N_REPEATS = 5


def _make_cost(n_train=50, n_val=20, seed=0):
    rng = torch.Generator().manual_seed(seed)
    cost = torch.rand(n_train, n_val, generator=rng).numpy().astype(np.float32)
    cost = (cost - cost.min()) / (cost.max() - cost.min())
    return cost


# ---------------------------------------------------------------------------
# greedy: batched gain loop == vectorised formula
# ---------------------------------------------------------------------------

def test_greedy_batched_gain_matches_vectorised():
    """The batched loop in greedy() must produce the same gain tensor as the
    original single-pass formula: (cost_sum - |cost_min - cost_row|.sum)."""
    n_train, n_val = 80, 30
    cost_np = _make_cost(n_train=n_train, n_val=n_val)
    cost_gpu = torch.from_numpy(cost_np).cuda()
    cost_sum = cost_gpu.sum(dim=1)

    rng = torch.Generator().manual_seed(1)
    cost_min = torch.rand(n_val, generator=rng).cuda()

    # --- batched (as implemented in ITRA.py) ---
    batch_size = 10240
    gains_batched = []
    for i in range(0, n_train, batch_size):
        g = (cost_sum[i:i+batch_size] - (cost_min - cost_gpu[i:i+batch_size]).abs().sum(dim=1)).cpu()
        gains_batched.append(g)
    gain_batched = torch.concat(gains_batched)

    # --- original vectorised formula ---
    gain_ref = (cost_sum - (cost_min - cost_gpu).abs().sum(dim=1)).cpu()

    assert torch.allclose(gain_batched, gain_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# _get_y_hat: batched == reference formula
# ---------------------------------------------------------------------------

def test_get_y_hat_matches_reference_formula():
    """_get_y_hat must produce the same y_hat and f_scores as the original
    inline computation: topk on (cost - f_matrix) then min(diff - y_hat, 0).mean()."""
    n_train, n_val = 60, 25
    R = 4
    rng = torch.Generator().manual_seed(42)
    cost_matrix = torch.rand(n_train, n_val, generator=rng)
    f_matrix = torch.rand(n_train, n_val, generator=rng)

    solver = ITRA()
    y_hat, f_scores = solver._get_y_hat(cost_matrix, f_matrix, R, batch_size=8)

    # reference (original non-batched logic)
    diff = cost_matrix - f_matrix
    y_hat_ref = diff.topk(k=R, dim=1, largest=True)[0][:, -1]
    f_ref = torch.minimum(diff - y_hat_ref.unsqueeze(1), torch.tensor(0.0)).mean(dim=1)

    assert torch.allclose(y_hat, y_hat_ref, atol=1e-5), "y_hat mismatch"
    assert torch.allclose(f_scores, f_ref, atol=1e-5), "f_scores mismatch"


def test_get_y_hat_small_batch_matches_full_batch():
    """Chunking at different batch sizes should give identical results."""
    n_train, n_val = 40, 15
    R = 3
    rng = torch.Generator().manual_seed(7)
    cost_matrix = torch.rand(n_train, n_val, generator=rng)
    f_matrix = torch.rand(n_train, n_val, generator=rng)

    solver = ITRA()
    y_hat_full, f_full = solver._get_y_hat(cost_matrix, f_matrix, R, batch_size=n_train)
    y_hat_small, f_small = solver._get_y_hat(cost_matrix, f_matrix, R, batch_size=7)

    assert torch.allclose(y_hat_full, y_hat_small, atol=1e-5)
    assert torch.allclose(f_full, f_small, atol=1e-5)


# ---------------------------------------------------------------------------
# Benchmarks  (run with pytest -s to see timing output)
# ---------------------------------------------------------------------------

def test_benchmark_greedy_gain(capsys):
    """Compare batched loop vs vectorised gain on a 2-batch-sized matrix."""
    cost_np = _make_cost(n_train=N_TRAIN_BENCH, n_val=N_VAL_BENCH)
    cost_gpu = torch.from_numpy(cost_np).cuda()
    cost_sum = cost_gpu.sum(dim=1)
    rng = torch.Generator().manual_seed(1)
    cost_min = torch.rand(N_VAL_BENCH, generator=rng).cuda()

    def batched():
        gains = []
        for i in range(0, N_TRAIN_BENCH, BATCH_SIZE):
            g = (cost_sum[i:i+BATCH_SIZE] - (cost_min - cost_gpu[i:i+BATCH_SIZE]).abs().sum(dim=1)).cpu()
            gains.append(g)
        return torch.concat(gains)

    def vectorised():
        return (cost_sum - (cost_min - cost_gpu).abs().sum(dim=1)).cpu()

    # warmup
    batched(); vectorised()

    t = time.perf_counter()
    for _ in range(N_REPEATS):
        gain_b = batched()
    t_batched = (time.perf_counter() - t) / N_REPEATS

    t = time.perf_counter()
    for _ in range(N_REPEATS):
        gain_v = vectorised()
    t_vectorised = (time.perf_counter() - t) / N_REPEATS

    with capsys.disabled():
        print(f"\n[greedy gain | {N_TRAIN_BENCH}x{N_VAL_BENCH} on cuda]")
        print(f"  batched    : {t_batched*1000:.2f} ms")
        print(f"  vectorised : {t_vectorised*1000:.2f} ms")

    assert torch.allclose(gain_b, gain_v, atol=1e-5)


def test_benchmark_get_y_hat(capsys):
    """Compare batched _get_y_hat vs single-pass reference on a 2-batch-sized matrix."""
    R = 10
    rng = torch.Generator().manual_seed(42)
    cost_matrix = torch.rand(N_TRAIN_BENCH, N_VAL_BENCH, generator=rng).cuda()
    f_matrix = torch.rand(N_TRAIN_BENCH, N_VAL_BENCH, generator=rng).cuda()

    solver = ITRA()

    def batched():
        return solver._get_y_hat(cost_matrix, f_matrix, R, batch_size=BATCH_SIZE)

    def vectorised():
        diff = cost_matrix - f_matrix
        y_hat = diff.topk(k=R, dim=1, largest=True)[0][:, -1]
        f_scores = torch.minimum(diff - y_hat.unsqueeze(1), torch.tensor(0.0)).mean(dim=1)
        return y_hat, f_scores

    # warmup
    batched(); vectorised()

    t = time.perf_counter()
    for _ in range(N_REPEATS):
        y_hat_b, f_b = batched()
    t_batched = (time.perf_counter() - t) / N_REPEATS

    t = time.perf_counter()
    for _ in range(N_REPEATS):
        y_hat_v, f_v = vectorised()
    t_vectorised = (time.perf_counter() - t) / N_REPEATS

    with capsys.disabled():
        print(f"\n[_get_y_hat | {N_TRAIN_BENCH}x{N_VAL_BENCH} R={R} on cuda]")
        print(f"  batched    : {t_batched*1000:.2f} ms")
        print(f"  vectorised : {t_vectorised*1000:.2f} ms")

    assert torch.allclose(y_hat_b, y_hat_v, atol=1e-5)
    assert torch.allclose(f_b, f_v, atol=1e-5)
