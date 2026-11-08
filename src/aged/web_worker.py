"""Isolated UI worker; the CLI and the web interface share the same trainer."""

import argparse
import json
from pathlib import Path
import time

from .config import Config
from .runtime import load_checkpoint
from .train import run_training

EVENT_PREFIX = "@@AGED_WEB@@"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    last_update = 0.0

    def progress(event):
        nonlocal last_update
        now = time.monotonic()
        if event["type"] == "batch" and now - last_update < 0.5 and event["batch"] != event["steps_per_epoch"]:
            return
        last_update = now
        print(EVENT_PREFIX + json.dumps(event, allow_nan=False), flush=True)

    if request.get("resume"):
        history = load_checkpoint(request["resume"])["history"]
        print(EVENT_PREFIX + json.dumps({"type": "history", "history": history}), flush=True)
    print("正在读取数据、拟合统计量并准备模型；首次加载可能需要一些时间。", flush=True)
    checkpoint = run_training(
        Config(**request["config"]), request["root"], request["out"],
        resume=request.get("resume"), init_from=request.get("init_from"),
        command_args=request["command_args"], progress_callback=progress,
        stop_requested=lambda: Path(request["stop_file"]).exists(),
    )
    print(f"Checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
