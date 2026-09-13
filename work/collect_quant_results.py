#!/usr/bin/env python3
"""Collect quantization supplement results from batch_run.log into a summary table.

Usage: python3 collect_quant_results.py <batch_run.log> [out.md] [out.tex]
"""
import re
import sys


METRICS = ["PSNR", "SSIM", "LPIPS", "DISTS", "CLIPIQA", "NIQE", "MUSIQ", "MANIQA", "FID"]
ORDER = ["minmax", "smoothquant", "viditq", "svd_full", "ours"]
LABELS = {
    "minmax": "Min-Max",
    "smoothquant": "SmoothQuant",
    "viditq": "ViDiT-Q",
    "svd_full": "SVD-full",
    "ours": "MobileTSD (Ours)",
}
PRECISIONS = ["w4a4", "w8a8", "w4a8", "w8a4"]


def prec_label(prec):
    return f"W{prec[1]}A{prec[3]}"


FP16 = {
    "PSNR": "24.79", "SSIM": "0.7171", "LPIPS": "0.2806", "DISTS": "0.2123",
    "CLIPIQA": "0.7035", "NIQE": "4.74", "MUSIQ": "69.78", "MANIQA": "0.6235",
    "FID": "118.08",
}


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
            mm = re.search(r"^PSNR: ([0-9.]+); SSIM: ([0-9.]+); LPIPS: ([0-9.]+); DISTS: ([0-9.]+); CLIPIQA: ([0-9.]+); NIQE: ([0-9.]+); MUSIQ: ([0-9.]+); MANIQA: ([0-9.]+) \| FID: ([0-9.]+)", lines[j])
            if mm:
                vals = [mm.group(k) for k in range(1, 10)]
                blocks[dname] = dict(zip(METRICS, vals))
                break
    return blocks


def fmt(v):
    f = float(v)
    if f >= 100:
        return f"{f:.2f}"
    if f >= 10:
        return f"{f:.2f}"
    return f"{f:.4f}"


def main():
    log_path = sys.argv[1]
    out_md = sys.argv[2] if len(sys.argv) > 2 else None
    out_tex = sys.argv[3] if len(sys.argv) > 3 else None
    blocks = parse_log(log_path)

    # Build table: rows (method, precision) -> metrics
    rows = []
    for meth in ORDER:
        for prec in PRECISIONS:
            key = f"{meth}_{prec}"
            if key in blocks:
                rows.append((meth, prec, blocks[key]))

    md = ["# W4A4 补充实验汇总（RealSR, 100 images）", ""]
    md.append("| 方法 | 精度 | " + " | ".join(METRICS) + " |")
    md.append("|---|:---:|" + "|".join("---:" for _ in METRICS) + "|")
    md.append(f"| w/o quant (FP16) | — | " + " | ".join(FP16[m] for m in METRICS) + " |")
    last_meth = None
    for meth, prec, vals in rows:
        label = LABELS[meth] if meth != last_meth else ""
        last_meth = meth
        md.append(f"| {label} | {prec_label(prec)} | " +
                  " | ".join(fmt(vals[m]) for m in METRICS) + " |")
    md.append("")
    md.append("> 注：Min-Max / SmoothQuant 均为 `--no_gptq` 严格口径；Ours 为 SVDQ-Global α=0.8（attn+5FFN，46 层）；SVD-full 为 dit_full（72 层）。")
    md_text = "\n".join(md)
    print(md_text)

    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text + "\n")
        print(f"\n[written] {out_md}")

    if out_tex:
        tex = [
            "\\begin{table*}[!t]",
            "\\setlength{\\abovecaptionskip}{0.1cm}",
            "\\caption{%",
            "Supplementary W4A4/W8A8/W4A8/W8A4 results on RealSR.",
            "``w/o quant.'' denotes the full-precision (FP16) baseline.",
            "}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{@{}l|ccccccccc@{}}",
            "\\toprule",
            "\\textbf{Method} & \\textbf{PSNR} $\\uparrow$ & \\textbf{SSIM} $\\uparrow$ & \\textbf{LPIPS} $\\downarrow$ & \\textbf{DISTS} $\\downarrow$ & \\textbf{CLIPIQA} $\\uparrow$ & \\textbf{NIQE} $\\downarrow$ & \\textbf{MUSIQ} $\\uparrow$ & \\textbf{MANIQA} $\\uparrow$ & \\textbf{FID} $\\downarrow$ \\\\",
            "\\midrule \\midrule",
            "w/o quant.                & 24.79 & 0.7171 & 0.2806 & 0.2123 & 0.7035 & 4.74 & 69.78 & 0.6235 & 118.08 \\\\",
            "\\midrule",
        ]
        last_meth = None
        for meth, prec, vals in rows:
            label = LABELS[meth] if meth != last_meth else ""
            if meth != last_meth and last_meth is not None:
                tex.append("\\cmidrule{2-10}")
            last_meth = meth
            tex.append(f"{label} ({prec_label(prec)})" + " & " +
                       " & ".join(fmt(vals[m]) for m in METRICS) + r" \\")
        tex += [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
        tex_text = "\n".join(tex)
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(tex_text + "\n")
        print(f"[written] {out_tex}")


if __name__ == "__main__":
    main()
