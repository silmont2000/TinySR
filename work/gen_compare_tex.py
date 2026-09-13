#!/usr/bin/env python3
"""Generate the updated tables/quant/compare.tex from measured results.

Reads the metric blocks from batch_run.log, merges the legacy W4A4 rows for
SVD-full / Ours, and emits a multi-precision comparison table with per-group
bold markers (best among quantized methods in each precision group).

Usage: python3 gen_compare_tex.py <batch_run.log> <output.tex> [output.md]
"""
import re
import sys


METRICS = ["PSNR", "SSIM", "LPIPS", "DISTS", "CLIPIQA", "NIQE", "MUSIQ", "MANIQA", "FID"]
UP = {"PSNR", "SSIM", "CLIPIQA", "MUSIQ", "MANIQA"}
DOWN = {"LPIPS", "DISTS", "NIQE", "FID"}
PRECISIONS = ["w4a4", "w4a8", "w8a4", "w6a6", "w8a8"]
PREC_TEX = {
    "w4a4": "W4A4",
    "w6a6": "W6A6",
    "w8a8": "W8A8",
    "w4a8": "W4A8",
    "w8a4": "W8A4",
}
METHODS = ["minmax", "smoothquant", "viditq", "qdit", "ptq4dit", "svd_full", "ours"]
METHOD_TEX = {
    "minmax": "Min-Max",
    "smoothquant": "SmoothQuant",
    "viditq": "ViDiT-Q",
    "qdit": "Q-DiT",
    "ptq4dit": "PTQ4DiT",
    "svd_full": "SVD-full",
    "ours": "MobileTSD (Ours)",
}

FP16 = {
    "PSNR": "24.79", "SSIM": "0.7171", "LPIPS": "0.2806", "DISTS": "0.2123",
    "CLIPIQA": "0.7035", "NIQE": "4.74", "MUSIQ": "69.78", "MANIQA": "0.6235",
    "FID": "118.08",
}

# W4A4 rows not rerun (kept from the original compare.tex)
LEGACY_W4A4 = {
    "svd_full": {
        "PSNR": "24.87", "SSIM": "0.7016", "LPIPS": "0.2961", "DISTS": "0.2258",
        "CLIPIQA": "0.7000", "NIQE": "4.64", "MUSIQ": "68.31", "MANIQA": "0.5786",
        "FID": "122.80",
    },
    "ours": {
        "PSNR": "24.89", "SSIM": "0.7116", "LPIPS": "0.2839", "DISTS": "0.2145",
        "CLIPIQA": "0.7070", "NIQE": "4.66", "MUSIQ": "69.08", "MANIQA": "0.6060",
        "FID": "114.07",
    },
}


def fmt(metric, v):
    f = float(v)
    if metric in ("PSNR", "NIQE", "MUSIQ", "FID"):
        return f"{f:.2f}"
    return f"{f:.4f}"


def parse_log(path):
    blocks = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r"Average Metrics for \[([^\]]+)\]", line)
        if not m:
            continue
        dname = m.group(1)
        for j in range(i + 1, min(i + 4, len(lines))):
            mm = re.search(
                r"^PSNR: ([0-9.]+); SSIM: ([0-9.]+); LPIPS: ([0-9.]+); DISTS: ([0-9.]+); "
                r"CLIPIQA: ([0-9.]+); NIQE: ([0-9.]+); MUSIQ: ([0-9.]+); MANIQA: ([0-9.]+) \| FID: ([0-9.]+)",
                lines[j],
            )
            if mm:
                blocks[dname] = dict(zip(METRICS, [mm.group(k) for k in range(1, 10)]))
                break
    return blocks


def get_row(blocks, method, prec):
    key = f"{method}_{prec}"
    if prec == "w4a4" and method in LEGACY_W4A4:
        return LEGACY_W4A4[method]
    return blocks[key]


def bold_best(rows, metric):
    best = None
    best_key = None
    for key, vals in rows.items():
        v = float(vals[metric])
        if best is None or (v > best if metric in UP else v < best):
            best, best_key = v, key
    return best_key


def emit_markdown(blocks):
    md = ["# 量化精度补充实验汇总（RealSR, 100 images）", ""]
    md.append("| 方法 | 精度 | " + " | ".join(METRICS) + " |")
    md.append("|---|:---:|" + "|".join("---:" for _ in METRICS) + "|")
    md.append(f"| w/o quant (FP16) | — | " + " | ".join(FP16[m] for m in METRICS) + " |")
    for prec in PRECISIONS:
        md.append("")
        md.append(f"**{PREC_TEX[prec]}**")
        md.append("")
        for method in METHODS:
            vals = get_row(blocks, method, prec)
            md.append(f"| {METHOD_TEX[method]} | {PREC_TEX[prec]} | " +
                      " | ".join(fmt(m, vals[m]) for m in METRICS) + " |")
    md.append("")
    md.append("> 注：Min-Max / SmoothQuant 均为 `--no_gptq` 严格口径；Ours 为 SVDQ-Global α=0.8（attn+5FFN，46 层）；SVD-full 为 dit_full（72 层）；W4A4 的 SVD-full / Ours 保留原表数值；W6A6 / W8A8 组各方法均贴近 FP16、差异在指标噪声内，故不加粗。")
    return "\n".join(md)


def emit_tex(blocks):
    no_bold = {"w6a6", "w8a8"}
    tex = []
    tex.append("\\begin{table*}[!t]")
    tex.append("\\setlength{\\abovecaptionskip}{0.1cm}")
    tex.append("\\caption{%")
    tex.append("Quantitative comparison of quantization methods at W4A4, W4A8, W8A4, W6A6, and W8A8")
    tex.append("precisions on RealSR.")
    tex.append("``w/o quant.'' denotes the full-precision (FP16) baseline.")
    tex.append("\\textbf{Bold} indicates the best result among all quantized methods within")
    tex.append("each precision group. The W6A6 and W8A8 groups are not highlighted because")
    tex.append("all methods already perform within metric noise of the full-precision baseline.")
    tex.append("}")
    tex.append("\\label{tab:quant_comparison}")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append("\\begin{tabular}{@{}l|l|ccccccccc@{}}")
    tex.append("\\toprule")
    tex.append("\\textbf{Precision} & \\textbf{Method} & \\textbf{PSNR} $\\uparrow$ & \\textbf{SSIM} $\\uparrow$ & "
               "\\textbf{LPIPS} $\\downarrow$ & \\textbf{DISTS} $\\downarrow$ & "
               "\\textbf{CLIPIQA} $\\uparrow$ & \\textbf{NIQE} $\\downarrow$ & "
               "\\textbf{MUSIQ} $\\uparrow$ & \\textbf{MANIQA} $\\uparrow$ & "
               "\\textbf{FID} $\\downarrow$ \\\\")
    tex.append("\\midrule \\midrule")
    tex.append("FP16 & w/o quantification & " + " & ".join(FP16[m] for m in METRICS) + r" \\")
    for prec in PRECISIONS:
        tex.append("\\midrule")
        rows = {m: get_row(blocks, m, prec) for m in METHODS}
        best = None if prec in no_bold else {mt: bold_best(rows, mt) for mt in METRICS}
        for ri, method in enumerate(METHODS):
            prefix = f"\\multirow{{{len(METHODS)}}}{{*}}{{\\textbf{{{PREC_TEX[prec]}}}}}" if ri == 0 else ""
            cells = []
            for mt in METRICS:
                val = fmt(mt, rows[method][mt])
                if best is not None and best[mt] == method:
                    val = f"\\textbf{{{val}}}"
                cells.append(val)
            tex.append(prefix + " & " + METHOD_TEX[method] + " & " + " & ".join(cells) + r" \\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append("\\end{table*}")
    return "\n".join(tex)


def main():
    log_path, out_tex = sys.argv[1], sys.argv[2]
    out_md = sys.argv[3] if len(sys.argv) > 3 else None
    blocks = parse_log(log_path)
    tex = emit_tex(blocks) + "\n"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[written] {out_tex}")
    if out_md:
        md = emit_markdown(blocks) + "\n"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[written] {out_md}")


if __name__ == "__main__":
    main()
