"""설치된 DINOv3와 제공 memory bank로 공개 split을 평가한다."""

import json
from pathlib import Path

import torch
from dinov3.hub import backbones
from industrial import paths


def main():
    config = paths.get_root_config()
    config.update(
        test_split=["test_public"],
        results_dir="/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/SuperADD/public_20261020",
    )
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["results_dir"], "config.json").write_text(json.dumps(config, indent=2))
    paths.get_root_config = lambda: config
    original_load = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov3":
            return getattr(backbones, model)(*args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    from industrial.test import main as test_main

    print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)
    print("설치된 DINOv3 사용; 공개 split; 정상 이미지 제외(원본 기본값)", flush=True)
    test_main()


if __name__ == "__main__":
    main()
