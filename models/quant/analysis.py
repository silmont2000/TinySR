import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from models.quant.layers import QuantLinearW4A4


class ActivationErrorAnalyzer:
    def __init__(self, max_samples_per_layer=200, max_points_per_sample=20000, seed=42):
        self.data = defaultdict(list)
        self.max_samples = max_samples_per_layer
        self.max_points = max_points_per_sample
        self.rng = np.random.default_rng(seed)
        self._last_input = {}
        self.handles = []

    def register_hooks(self, transformer):
        for name, m in transformer.named_modules():
            if not isinstance(m, QuantLinearW4A4):
                continue
            pre_handle = m.act_quantizer.register_forward_pre_hook(
                self._make_pre_hook(name))
            post_handle = m.act_quantizer.register_forward_hook(
                self._make_post_hook(name))
            self.handles.extend([pre_handle, post_handle])

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._last_input.clear()

    def _make_pre_hook(self, layer_name):
        def hook(module, input):
            self._last_input[layer_name] = input[0].detach().float().cpu()
        return hook

    def _make_post_hook(self, layer_name):
        def hook(module, input, output):
            x = self._last_input.pop(layer_name, None)
            x_q = output.detach().float().cpu()
            if x is None:
                return
            x_np = x.flatten().numpy()
            x_q_np = x_q.flatten().numpy()
            n = x_np.size
            if n > self.max_points:
                idx = self.rng.choice(n, size=self.max_points, replace=False)
                x_np = x_np[idx]
                x_q_np = x_q_np[idx]
            store = self.data[layer_name]
            if len(store) < self.max_samples:
                store.append((x_np, x_q_np))
        return hook

    def compute_report(self):
        report = {"layers": {}}
        for layer_name in sorted(self.data.keys()):
            samples = self.data[layer_name]
            if not samples:
                continue

            all_x = np.concatenate([s[0] for s in samples])
            all_x_q = np.concatenate([s[1] for s in samples])

            error = all_x - all_x_q
            mse = float(np.mean(error ** 2))
            mae = float(np.mean(np.abs(error)))
            max_abs_err = float(np.max(np.abs(error)))
            signal_power = float(np.mean(all_x ** 2))
            noise_power = mse
            snr = float(10.0 * math.log10(signal_power / noise_power)) if noise_power > 1e-12 else float("inf")
            mean_err = float(np.mean(error))
            std_err = float(np.std(error))

            n = all_x_q.size
            mu = float(np.mean(all_x_q))
            std = float(np.std(all_x_q))
            kurt = float(np.mean((all_x_q - mu) ** 4) / (std ** 4) - 3.0) if std > 1e-12 else 0.0
            outlier_ratio = float(np.mean(np.abs(all_x_q - mu) > 5.0 * std)) if std > 1e-12 else 0.0

            sample_kurts = []
            for _, x_q_s in samples:
                s_mu = x_q_s.mean()
                s_std = x_q_s.std()
                if s_std > 1e-12 and x_q_s.size >= 4:
                    sample_kurts.append(float(np.mean((x_q_s - s_mu) ** 4) / (s_std ** 4) - 3.0))
                else:
                    sample_kurts.append(0.0)
            avg_sample_kurt = float(np.mean(sample_kurts)) if sample_kurts else 0.0

            mu_x = float(np.mean(all_x))
            std_x = float(np.std(all_x))
            kurt_x = float(np.mean((all_x - mu_x) ** 4) / (std_x ** 4) - 3.0) if std_x > 1e-12 else 0.0

            report["layers"][layer_name] = {
                "n_total": int(n),
                "error": {
                    "mse": mse, "mae": mae, "max_abs_err": max_abs_err,
                    "snr": snr, "mean": mean_err, "std": std_err,
                },
                "quantized": {
                    "mean": mu, "std": std, "kurtosis_excess": kurt,
                    "outlier_ratio_5sigma": outlier_ratio,
                    "avg_per_sample_kurtosis": avg_sample_kurt,
                },
                "unquantized": {
                    "mean": mu_x, "std": std_x, "kurtosis_excess": kurt_x,
                },
            }
        return report

    def save_report(self, path, args_dict=None):
        report = self.compute_report()
        if args_dict:
            report["args"] = args_dict
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Analyzer] activation error report -> {path}")
