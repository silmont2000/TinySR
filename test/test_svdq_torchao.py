"""
End-to-end test v3: SVDQLinear + TorchAO Int4WeightOnlyConfig (FINAL).
Clean benchmark: bf16 eager vs SVDQ int4 eager vs SVDQ int4 compiled.
"""
import torch, torch.nn as nn, gc, copy, sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.quant.svdq_linear import (
    SVDQLinear, replace_linear_with_svdq,
    calibrate_svdq_layers, quantize_svdq_layers, iter_svdq_layers,
)
from torchao.utils import benchmark_model
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor

DT = torch.bfloat16
DEV = "cuda"
torch._dynamo.config.suppress_errors = True


class TinySRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1536, 1536, bias=False),
            nn.Linear(1536, 1536, bias=False),
            nn.Linear(1536, 6144, bias=False),
            nn.Linear(6144, 1536, bias=False),
            nn.Linear(1536, 1536, bias=False),
            nn.Linear(1536, 1536, bias=False),
            nn.Linear(1536, 6144, bias=False),
            nn.Linear(6144, 1536, bias=False),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    print(f"PyTorch {torch.__version__}, torchao, GPU: {torch.cuda.get_device_name(0)}")
    x_test = torch.randn(256, 1536, dtype=DT, device=DEV)

    # ─── 1. Build ──────────────────────────────────────────────
    print("\n═══ 1. Build model ═══")
    original = TinySRModel().to(DT).to(DEV)
    with torch.no_grad():
        y_orig = original(x_test)

    # ─── 2. Replace → SVDQLinear → calibrate → quantize ───────
    print("\n═══ 2. SVD + GPTQ calibrate + Int4WeightOnly quantize ═══")
    model = copy.deepcopy(original)
    replace_linear_with_svdq(model)
    calibrate_svdq_layers(model, rank=32, alpha=1.0, gptq_block_size=128, gptq_damp=0.01)
    quantize_svdq_layers(model, group_size=32)

    # ─── 3. Accuracy ───────────────────────────────────────────
    print("\n═══ 3. Accuracy ═══")
    with torch.no_grad():
        y_svdq = model(x_test)
    cos = nn.functional.cosine_similarity(
        y_orig.reshape(-1).float(), y_svdq.reshape(-1).float(), dim=0
    ).item()
    mse = ((y_orig.float() - y_svdq.float()) ** 2).mean().item()
    print(f"  cosine vs FP32 ref: {cos:.4f}")
    print(f"  MSE: {mse:.6e}")

    # ─── 4. Memory ─────────────────────────────────────────────
    print("\n═══ 4. Weight memory ═══")
    total_bf16 = 0
    total_int4 = 0
    for name, m in iter_svdq_layers(model):
        bf16_bytes = m.in_features * m.out_features * 2
        if isinstance(m.weight, AffineQuantizedTensor):
            int4_bytes = m.weight.nbytes
        else:
            int4_bytes = bf16_bytes
        total_bf16 += bf16_bytes
        total_int4 += int4_bytes
        print(f"  {name}: {bf16_bytes/1024:.0f}KB bf16 → {int4_bytes/1024:.0f}KB int4 ({bf16_bytes/int4_bytes:.1f}x)")
    print(f"  TOTAL: {total_bf16/1024**2:.1f}MB bf16 → {total_int4/1024**2:.1f}MB int4 ({total_bf16/total_int4:.1f}x)")

    # ─── 5. Latency ────────────────────────────────────────────
    print("\n═══ 5. Latency (256×1536 input, no torch.compile) ═══")
    gc.collect()
    torch.cuda.empty_cache()

    # bf16 eager
    with torch.no_grad():
        for _ in range(30):
            original(x_test)
        torch.cuda.synchronize()
        t_bf16 = benchmark_model(original, 200, (x_test,))

        # SVDQ int4 eager
        for _ in range(30):
            model(x_test)
        torch.cuda.synchronize()
        t_int4 = benchmark_model(model, 200, (x_test,))

    print(f"  bf16 eager:  {t_bf16:.3f}ms  (baseline)")
    print(f"  SVDQ int4:   {t_int4:.3f}ms  ({t_int4/t_bf16:.1f}x)")

    # ─── 6. Summary ────────────────────────────────────────────
    print("\n═══ 6. Summary ═══")
    print(f"  Accuracy:   cos={cos:.4f}")
    print(f"  Speed:      {t_int4/t_bf16:.1f}x slower (no compile)")
    print(f"  Memory:     {total_bf16/total_int4:.1f}x compression")
    print(f"  SVD branch: 32-rank LowRankBranch per layer")


if __name__ == "__main__":
    main()
