"""Run the existing suite on CPU, rejecting network connections and downloads."""

import os
import socket
import sys
from pathlib import Path

os.environ.update(
    # "-1" hides every device; an empty string still reports CUDA as available on
    # a driver-equipped machine, which made this guard fail locally (2026-11-17).
    CUDA_VISIBLE_DEVICES="-1",
    GRADIO_ANALYTICS_ENABLED="False",
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
)
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def deny_network(*args, **kwargs):
    raise RuntimeError("CPU checks are offline; prepare LPIPS weights before running")


def main():
    import torch

    weight = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if not weight.is_file():
        raise SystemExit(
            "Missing AlexNet weights. Run: python -c "
            '\"import lpips; lpips.LPIPS(net=\'alex\', verbose=False)\"'
        )
    if torch.cuda.device_count():
        raise SystemExit(
            f"CPU check sees {torch.cuda.device_count()} CUDA device(s); "
            "CUDA_VISIBLE_DEVICES=-1 did not take effect"
        )
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    torch.hub.download_url_to_file = deny_network

    import pytest

    return pytest.main(["tests", "-q", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
