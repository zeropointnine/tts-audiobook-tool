"""
Disposable diagnostic: sample the live model-worker's GPU + host memory while
the app generates, to see whether VRAM grows monotonically per generation.

Non-invasive: reads NVML and /proc only. Never touches the app, never loads a
model, never allocates GPU memory.

Run:
    ./venv-cb/bin/python chatterbox_turbo_vram_probe.py

Output: CSV at /tmp/turbo_vram_probe.csv plus one compact line per sample.
Stop with Ctrl-C.
"""

import csv
import os
import sys
import time

import psutil
import pynvml

CSV_PATH = "/tmp/turbo_vram_probe.csv"
INTERVAL_SECONDS = 2.0
WORKER_LOG = "/tmp/tts-audiobook-tool-worker.log"

GB = 1024 ** 3


def find_worker_pid() -> int | None:
    """The model worker is the multiprocessing-spawn child of the app process."""
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "spawn_main" not in cmd or "--multiprocessing-fork" not in cmd:
                continue
            parent = psutil.Process(p.info["ppid"])
            parent_cmd = " ".join(parent.cmdline() or [])
            if "tts_audiobook_tool" in parent_cmd:
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def nvml_snapshot():
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used), int(info.total)
    finally:
        pynvml.nvmlShutdown()


def nvml_process_bytes(pid: int) -> int | None:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid and proc.usedGpuMemory is not None:
                return int(proc.usedGpuMemory)
        return None
    except pynvml.NVMLError:
        return None
    finally:
        pynvml.nvmlShutdown()


def generation_count() -> int:
    try:
        with open(WORKER_LOG, "r", errors="replace") as f:
            return sum(1 for line in f if "generate_using_project dispatch" in line)
    except OSError:
        return -1


def main() -> int:
    pynvml.nvmlInit()
    pynvml.nvmlShutdown()

    new_file = not os.path.exists(CSV_PATH)
    out = open(CSV_PATH, "a", newline="")
    writer = csv.writer(out)
    if new_file:
        writer.writerow([
            "time", "worker_pid", "event", "gens_cumulative",
            "proc_vram_gb", "device_used_gb", "device_free_gb", "worker_rss_gb",
        ])

    last_pid: int | None = None
    print(f"Sampling every {INTERVAL_SECONDS}s -> {CSV_PATH}", flush=True)

    while True:
        pid = find_worker_pid()
        event = ""
        if pid is None:
            event = "no-worker"
        elif pid != last_pid:
            event = "worker-start" if last_pid is None else "worker-restart"
            last_pid = pid

        proc_vram = nvml_process_bytes(pid) if pid else None
        dev_used, dev_total = nvml_snapshot()
        rss = None
        if pid:
            try:
                rss = psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass

        row = [
            time.strftime("%H:%M:%S"),
            pid if pid else "",
            event,
            generation_count(),
            f"{proc_vram / GB:.3f}" if proc_vram else "",
            f"{dev_used / GB:.3f}",
            f"{(dev_total - dev_used) / GB:.3f}",
            f"{rss / GB:.3f}" if rss else "",
        ]
        writer.writerow(row)
        out.flush()
        print(" ".join(str(c) for c in row), flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
