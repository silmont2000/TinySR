import torch
import torch.nn as nn

from models.quant.layers import QuantLinearW4A4, iter_quant_layers


@torch.no_grad()
def load_quantized_model_state(transformer: nn.Module, path: str, device=None):
    data = torch.load(path, map_location=device or "cpu")
    missing, unexpected = transformer.load_state_dict(
        data["state_dict"], strict=False)
    if missing:
        print(
            f"[W4A4] load: missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(
            f"[W4A4] load: unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    print(f"[W4A4] quantized model state loaded <- {path}")
    return data.get("replaced_layers"), data.get("quant_meta", []), data.get("model_args", {})
