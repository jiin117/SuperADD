"""기존 memory bank로 공개 정상 이미지만 추가 추론한다."""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from dinov3.hub import backbones
from industrial.dataset import MVTecAD2Dataset
from industrial.model import SuperADD
from industrial.paths import get_dataset_path, get_model_path, get_root_config
from torchvision.transforms import ToTensor
from tqdm import tqdm

OUTPUT = Path('/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/SuperADD/good_20261021')


def main():
    config = get_root_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'config.json').write_text(json.dumps(dict(
        original_config=config, split='test_public/good',
        backbone_loader='installed dinov3.hub.backbones',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0)), indent=2))
    original_load = torch.hub.load

    def load(repo, model, *args, **kwargs):
        if repo == 'facebookresearch/dinov3':
            return getattr(backbones, model)(*args, **kwargs)
        return original_load(repo, model, *args, **kwargs)

    torch.hub.load = load
    for category in config['categories']:
        dataset = MVTecAD2Dataset(get_dataset_path('mvtec_ad_2'), category, 'test_public', ToTensor())
        dataset.samples = [sample for sample in dataset.samples if sample.label == 0]
        if not dataset.samples:
            raise RuntimeError(f'정상 이미지가 없습니다: {category}')
        mask_dir = OUTPUT / 'anomaly_images_thresholded' / category / 'test_public/good'
        map_dir = OUTPUT / 'anomaly_images' / category / 'test_public/good'
        mask_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)
        print(category, 'normal images', len(dataset), flush=True)
        model = SuperADD.from_disk(get_model_path(category))
        for sample in tqdm(dataset, desc=category):
            anomaly, binary = model.predict(sample.image)
            if not np.isfinite(anomaly).all():
                raise ValueError(f'유한하지 않은 예측값: {sample.image_path}')
            stem = Path(sample.image_path).stem
            tifffile.imwrite(map_dir / f'{stem}.tiff', anomaly.astype(np.float16))
            if not cv2.imwrite(str(mask_dir / f'{stem}.png'), binary):
                raise OSError(f'마스크 저장 실패: {stem}')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print('정상 이미지 추론 완료', flush=True)


if __name__ == '__main__':
    main()
