#!/bin/bash
# Wait for the main supplement batch to finish, then run the 3 W4A4 consistency
# reruns (Min-Max / SmoothQuant with --no_gptq, ViDiT-Q with current code).
while pgrep -f "[r]un_quant_supplement.sh" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date '+%F %T')] main batch finished, starting W4A4 reruns" > /root/quant_extra/w4a4_rerun_stdout.log
cd /root/quant_extra
ONLY_W4A4=1 bash /root/quant_extra/run_quant_supplement.sh >> /root/quant_extra/w4a4_rerun_stdout.log 2>&1
echo "[$(date '+%F %T')] W4A4 reruns done" >> /root/quant_extra/w4a4_rerun_stdout.log
