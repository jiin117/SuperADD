# SuperADD

> 공통 규칙은 루트 `../AGENTS.md`를 따른다. 이 문서는 SuperADD 전용 규칙이다.

## 1. 개요

- 원본: https://github.com/LukasRoom/SuperADD (클론 시점 커밋 `44cf251`, 2026-05-18)
- 논문: arXiv 2605.14808 — *SuperADD: Training-free Class-agnostic Anomaly Segmentation* (CVPR 2026 VAND 4.0 Industrial Track)
- 방식: 정상 이미지만 사용하는 class-agnostic 이상 탐지. SuperAD 기반이며 DINOv3 backbone을 쓴다.
- "학습"은 가중치를 학습하지 않는다. 카테고리별 memory bank(`.npz`)와 설정(`.json`)을 만드는 과정이다.
- 라이선스: `LICENSE`, `NOTICE`, `third_party_licenses/` 참고
- 전체 연산·수식·코드 비교: [SuperADD·BALM 알고리즘 상세](../documents/SuperADD_BALM_알고리즘_상세.md)

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
| 2026-10-21 | `run_reproduce.py`, `run_good.py` (신규) | 설치된 DINOv3로 공개 이상/정상 이미지를 별도 결과 폴더에 추론하는 래퍼 | 원본 코드를 유지하며 공개 평가 및 정상 이미지 오탐률 계산 |

## 7. 진행 상태 (1단계)

- [x] `uv sync --all-packages` 성공 (2026-10-16, torch 2.11.0+cu128, B200 CUDA 동작 확인)
- [x] DINOv3 가중치 준비 (2026-10-20, `to/weights/SuperADD/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`, `weights/`에 링크)
- [x] MVTec AD 2 준비 (2026-10-16, `data/mvtec_ad_2` → `to/data/mvtec_ad_2` 링크, 8개 카테고리·전 split 확인)
- [x] 사전 생성 모델 준비 (2026-10-16, `models_submission08` 16.4GB → `to/weights/SuperADD/models/`, `models/` 안에 파일별 링크)
- [ ] 추론 재현 (사전 생성 모델 사용)

### 공개 split 추론 실행 (2026-10-20)

- `source ../env.sh` 후 `CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync python run_reproduce.py`로 실행.
- 로컬 추가 파일 `run_reproduce.py`는 원본 추론 코드를 호출하되, `test_split`을 `test_public`으로 바꾸고 결과를 `to/results/SuperADD/public_20261020/`에 저장한다. 원본 `config.json`은 유지한다.
- DINOv3는 GitHub 최신 코드를 받는 대신 기존 uv 환경에 설치된 `dinov3.hub.backbones`로 로딩한다. 제공 memory bank와 후처리·정상 이미지 제외 설정은 원본 그대로 사용한다.
- 환경: torch 2.11.0+cu128, NVIDIA B200. 로그: 위 결과 폴더의 `inference.log`.
- 공개 split 8개 카테고리 추론 완료. 정상 이미지를 제외한 705장에 TIFF·이진 마스크 각 705개 생성. `scores_test_public.csv`의 평균 AU는 0.83929238, F1은 0.62612381. AU는 원본 코드의 `roc_auc_score(max_fpr=0.05)` 값이다.
- `can` AU 0.51609888, F1 0.0으로 재현 검증은 미완료. 예측값은 유한하며 범위 0.05185~1.18359, 이진 마스크 양성 픽셀 57,699개를 확인했다. 원인과 원본의 동일 조건 점수 대비 차이는 미정.

### can 재점검 (2026-10-21)

- 90개 예측 TIFF와 정답을 파일명으로 대응시켜 재계산: 평가 해상도 `(256, 558)`에서 정답 양성 1,194픽셀, 예측 양성 57,699픽셀, 교집합 0픽셀. F1 0은 실제 출력과 일치한다.
- 정답 영역의 저장된 이상 점수 최대는 0.74121094로 제공 임계값 0.79266065보다 낮다. 전체 ROC AUC는 0.75994889이며 기존 표의 AU 0.51609888은 `max_fpr=0.05`로 계산한 표준화 부분 AUC다.
- 정답 파일 누락·해상도 불일치는 확인되지 않았다. `000_regular`의 원본 정답 결함은 60픽셀(1024×2232 이미지), 기존 평가 방식으로 축소한 뒤 6픽셀이다.
- `000_regular`, `007_regular` 두 장을 같은 환경에서 재추론: 저장된 float16 anomaly map과 최대 절대 차이 0, 이진 마스크 완전 일치. 실행의 비결정성이나 저장 오류로 인한 현상은 이 두 장에서 재현되지 않았다.
- 근본적인 저성능 원인(미세 결함에 대한 특징·해상도·memory bank 적합성 등)은 미정. 공개 정답에 맞춰 임계값을 조정하지 않았으며 원본 대비 재현 성공 판정은 보류한다.

### 정상 이미지 추가 추론 (2026-10-21)

- `source ../env.sh` 후 `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run --locked --no-sync python run_good.py` 실행. 제공 memory bank·카테고리별 threshold·후처리는 기존 이상 이미지 추론과 같다.
- `test_public/good` 379장이 대상이며, `to/results/SuperADD/good_20261021/`에 이진 마스크·float16 점수맵·설정·로그를 저장한다. 기존 이상 이미지 결과는 덮어쓰지 않는다.
- 정상 데이터만 사용하면 원본 `test.py`의 ROC AUC 계산이 단일 클래스에서 실패하므로, 원본 `SuperADD.predict()`만 호출하는 전용 루프를 사용한다.
- 379장 추가 추론과 오탐률 계산 완료. 원본 최종 마스크에 양성 픽셀이 1개 이상인 정상 이미지는 121/379장(31.9261%), 공통 1/4 해상도의 합산 픽셀 FPR은 0.0936748%. `can`은 29/72장(40.2778%). [두 모델 비교 및 기준](../GOOD_FPR_COMPARISON.md)

### 논문 공개 기준 수치 확인 (2026-10-22)

- [논문 표 1](https://arxiv.org/pdf/2605.14808)은 공개 `can` F1 0.00%, 평균 F1 62.61%, 평균 AU-ROC0.05 83.93%를 보고한다. 이번 공개 평균은 소수 둘째 자리까지 일치하며, `can` F1 0 자체는 실행 실패를 뒷받침하지 않는다. `fruit_jelly`, `vial` 등 일부 카테고리의 작은 수치 차이는 남아 있다. 세부 비교는 위 알고리즘 상세 문서 9절을 참고한다.
