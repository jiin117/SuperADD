# SuperADD

> 공통 규칙은 루트 `../AGENTS.md`를 따른다. 이 문서는 SuperADD 전용 규칙이다.

## 1. 개요

- 원본: https://github.com/LukasRoom/SuperADD (클론 시점 커밋 `44cf251`, 2026-05-18)
- 논문: arXiv 2605.14808 — *SuperADD: Training-free Class-agnostic Anomaly Segmentation* (CVPR 2026 VAND 4.0 Industrial Track)
- 방식: 정상 이미지만 사용하는 class-agnostic 이상 탐지. SuperAD 기반이며 DINOv3 backbone을 쓴다.
- "학습"은 가중치를 학습하지 않는다. 카테고리별 memory bank(`.npz`)와 설정(`.json`)을 만드는 과정이다.
- 라이선스: `LICENSE`, `NOTICE`, `third_party_licenses/` 참고

## 2. 환경 (uv)

- 원본이 **처음부터 uv workspace** 구조다. 별도 변환 없이 원본 `pyproject.toml`과 `uv.lock`을 그대로 쓴다.
  - 루트 `pyproject.toml`: workspace (`utils`, `tracks/industrial`)
  - `requires-python >= 3.11`
  - torch ≥ 2.11, torchvision 0.26.0, `pytorch-cu128` 인덱스 → **B200 호환 예상**
  - `dinov3`는 GitHub(`facebookresearch/dinov3`)에서 설치된다.
- 설치:
  ```bash
  cd SuperADD
  uv sync --all-packages
  ```

## 3. 데이터 및 가중치

| 항목 | 기본 경로 | 비고 |
|---|---|---|
| MVTec AD 2 | `./data/mvtec_ad_2` | `config.json`의 `datasets_dir` |
| DINOv3 가중치 | `./weights/` | Meta 공식 사이트에서 신청 후 다운로드. `dino_weights_dir` |
| 사전 생성 모델(memory bank) | `./models/` | Fraunhofer ownCloud 링크(README 참고). `models_dir` |
| 결과 | `./outputs/` | `results_dir` |
| 제출 파일 | `./submissions/` | `submissions_dir` |

- backbone: `dinov3_vith16plus`, layers `[7, 15, 23, 31]`
- 카테고리: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts

## 4. 실행

모든 설정은 `config.json`에 있다. **설정을 바꾸면 train → test → submit을 다시 실행한다.**

```bash
uv run train-industrial    # 카테고리별 memory bank 생성
uv run test-industrial     # anomaly map, thresholded map 생성
uv run submit-industrial   # 제출 파일 생성
```

- 기본 `test_split`은 `["test_private", "test_private_mixed"]`이다. `test_public`으로 평가하려면 `config.json`에 추가한다.
- 사전 생성 모델을 `./models/`에 두면 train을 건너뛰고 바로 test를 실행할 수 있다(DINOv3 가중치는 여전히 필요).

## 5. 주의사항

- `data/`, `weights/`, `models/`는 비어 있는 상태로 클론된다. 내용물은 커밋하지 않는다.
- `config.json`을 수정하면 수정 내역을 아래 "로컬 변경 내역"에 기록한다.

## 6. 로컬 변경 내역

| 날짜 | 파일 | 내용 | 이유 |
|---|---|---|---|
| 2026-10-16 | `pyproject.toml` | `[tool.uv]`에 `cache-dir = "/NHNHOME/WORKSPACE/26moel002_ex07/AD/AD_robustness/.uv-cache"` 추가 | 공통 uv 캐시 사용 (루트 디스크 용량 부족) |
| 2026-10-16 | `pyproject.toml` | `cache-dir`을 `/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/.uv-cache`로 변경. `.venv`는 `to/venvs/SuperADD` 링크 | 프로젝트를 `/home/daegu`로 이동, 대용량은 `to/`로 분리 |

## 7. 진행 상태 (1단계)

- [x] `uv sync --all-packages` 성공 (2026-10-16, torch 2.11.0+cu128, B200 CUDA 동작 확인)
- [ ] DINOv3 가중치 준비
- [ ] MVTec AD 2 준비
- [ ] 추론 재현 (사전 생성 모델 사용)
