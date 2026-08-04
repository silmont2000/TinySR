"""Unit tests for script/analyze_quant_sensitivity.py pure-math helpers.

Run:  python -m pytest test/test_quant_sensitivity.py -q
  or:  python test/test_quant_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from analyze_quant_sensitivity import (
    moment_stats,
    outlier_ratio,
    layer_salience,
    predicted_wquant_out_mse,
    StreamingActStats,
)


def _np_excess_kurtosis(x):
    x = x.astype(np.float64)
    mu = x.mean()
    var = x.var()
    mu4 = ((x - mu) ** 4).mean()
    return mu4 / (var ** 2) - 3.0


def test_moment_stats_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000) * 2.0 + 1.0
    s1 = x.sum(); s2 = (x ** 2).sum(); s3 = (x ** 3).sum(); s4 = (x ** 4).sum()
    mean, var, kurt = moment_stats(s1, s2, s3, s4, x.size)
    assert abs(mean - x.mean()) < 1e-6
    assert abs(var - x.var()) < 1e-4
    assert abs(kurt - _np_excess_kurtosis(x)) < 5e-3


def test_moment_stats_heavy_tail_positive_kurtosis():
    rng = np.random.default_rng(1)
    # Laplace has excess kurtosis 3 > 0; add a few extreme outliers
    x = rng.laplace(size=100_000)
    x[:5] = 50.0
    s1 = x.sum(); s2 = (x ** 2).sum(); s3 = (x ** 3).sum(); s4 = (x ** 4).sum()
    _, _, kurt = moment_stats(s1, s2, s3, s4, x.size)
    assert kurt > 3.0


def test_outlier_ratio():
    assert outlier_ratio(10.0, 2.0) == 5.0
    assert outlier_ratio(1.0, 0.0) > 1e6  # eps guard, no div-by-zero


def test_streaming_matches_numpy_chunked():
    rng = np.random.default_rng(2)
    C = 8
    x = rng.standard_normal((5000, C)).astype(np.float32) * 1.5
    st = StreamingActStats(C, reservoir_cap=10_000_000)  # cap >> n => exact
    xt = torch.from_numpy(x)
    for chunk in torch.split(xt, 700):        # feed in uneven chunks
        st.update(chunk)
    # per-channel E[x^2]
    ex2 = st.e_x2().numpy()
    np.testing.assert_allclose(ex2, (x ** 2).mean(axis=0), rtol=1e-5, atol=1e-5)
    fin = st.finalize()
    flat = x.reshape(-1)
    assert abs(fin["act_absmax"] - np.abs(flat).max()) < 1e-4
    assert abs(fin["act_excess_kurtosis"] - _np_excess_kurtosis(flat)) < 5e-3
    # reservoir holds all elements => exact P99.9
    exp_p999 = np.quantile(np.abs(flat), 0.999)
    assert abs(fin["act_p999"] - exp_p999) < 1e-2
    assert fin["n_tokens"] == 5000


def test_streaming_reservoir_p999_approx():
    rng = np.random.default_rng(3)
    x = np.abs(rng.standard_normal((20000, 4)).astype(np.float32))
    st = StreamingActStats(4, reservoir_cap=50_000, seed=7)
    st.update(torch.from_numpy(x))
    fin = st.finalize()
    exp = np.quantile(np.abs(x.reshape(-1)), 0.999)
    # subsampled reservoir: within 15% of true P99.9
    assert abs(fin["act_p999"] - exp) / exp < 0.15


def test_layer_salience_formula():
    W = torch.tensor([[1.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]])
    h = torch.tensor([1.0, 4.0, 2.0])       # E[x_j^2]
    out = layer_salience(W, h)
    # S = W^2 * h = [[1*1, 4*4, 0],[0,0,9*2]] = [[1,16,0],[0,0,18]]
    assert abs(out["salience_total"] - (1 + 16 + 18)) < 1e-6
    assert abs(out["salience_mean"] - 35.0 / 6.0) < 1e-6
    # channel salience col = [1, 16, 18]; top-1 channel (top1% -> at least 1) = 18
    assert abs(out["salience_top1pct_ch"] - 18.0 / 35.0) < 1e-6


def test_layer_salience_matches_bruteforce():
    torch.manual_seed(0)
    W = torch.randn(64, 48)
    h = torch.rand(48) + 0.1
    out = layer_salience(W, h)
    S = (W ** 2) * h.reshape(1, -1)
    assert abs(out["salience_total"] - float(S.sum())) < 1e-3
    k = max(1, int(S.numel() * 0.01))
    exp_top = float(S.reshape(-1).topk(k).values.sum() / S.sum())
    assert abs(out["salience_top1pct_w"] - exp_top) < 1e-6


def test_predicted_wquant_mse_manual():
    # single group, int4 (qmax=7). W constant magnitude => absmax known.
    W = torch.tensor([[4.0, -4.0, 2.0, 4.0]])   # in=4, group_size=4 => one group
    h = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mse, rel = predicted_wquant_out_mse(W, h, group_size=4, bits=4)
    absmax = 4.0
    delta = absmax / 7.0
    errvar = delta ** 2 / 12.0
    exp_mse = errvar * 4          # 4 weights, h=1
    assert abs(mse - exp_mse) < 1e-6
    signal = float((W ** 2).sum())
    assert abs(rel - exp_mse / signal) < 1e-6


def test_predicted_wquant_relative_scales_with_activation():
    torch.manual_seed(1)
    W = torch.randn(32, 64)
    h1 = torch.ones(64)
    h2 = torch.ones(64) * 5.0
    _, rel1 = predicted_wquant_out_mse(W, h1)
    _, rel2 = predicted_wquant_out_mse(W, h2)
    # relative NMSE is scale-invariant in h (both mse and signal scale by h)
    assert abs(rel1 - rel2) < 1e-5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
