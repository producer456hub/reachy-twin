"""Live compute telemetry for the panel: GPU (nvidia-smi) + NPU (Lemonade status).

Both degrade gracefully to {"available": False} on hosts that lack them (e.g. the
Mac has no nvidia-smi and no AMD NPU), so the panel simply hides those meters.

NOTE on the NPU: AMD's XDNA NPU exposes no live utilization % on Windows (no perf
counter, no smi), so we report a busy/idle signal from Lemonade's loaded model
instead of a percentage.
"""
import json
import shutil
import subprocess
import urllib.request


def gpu_stats():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        row = out.stdout.strip().splitlines()[0].split(",")
        name, util, used, total = [x.strip() for x in row]
        return {"available": True, "name": name, "util": int(util),
                "mem_used": int(used), "mem_total": int(total)}
    except Exception:
        return {"available": False}


def npu_stats(lemonade_url):
    try:
        with urllib.request.urlopen(lemonade_url.rstrip("/") + "/api/v0/health", timeout=3) as r:
            h = json.loads(r.read())
        loaded = h.get("model_loaded")
        return {"available": True, "active": bool(loaded), "model": loaded}
    except Exception:
        return {"available": False}     # Lemonade not running / no NPU on this host
