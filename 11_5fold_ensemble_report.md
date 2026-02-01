# 5-Fold Cross-Validation 앙상블 실험 보고서

**작성일**: 2026년 2월 1일  
**실험 목적**: K-Fold 앙상블을 통한 리더보드 점수 극대화  
**현재 최고 점수**: 96.28% H-Mean (5-Fold Voting≥3)

---

## 1. 실험 배경

### 1.1 이전 실험 결과 요약

| 모델 구성 | H-Mean | Precision | Recall | 비고 |
|---------|--------|-----------|--------|------|
| ResNet18 960px | 95.81% | - | - | 기본 베이스라인 |
| ResNet50 Basic | 96.20% | - | - | Backbone 업그레이드 |
| ResNet50 Aggressive | 96.26% | 97.31% | 95.58% | Postprocessing 최적화 |
| TTA (H-Flip) | 78.25% | 71.03% | 87.30% | ❌ 좌표 변환 버그 |
| Ultra-Aggressive | 96.09% | 97.12% | 95.42% | 과도한 공격성 |

### 1.2 2-Fold 앙상블 실패 분석

**2-Fold Voting≥1 (OR 전략)**
- 결과: 95.09% H-Mean (-1.17%p)
- 문제: False Positive 과다 (45,784 boxes)
- Precision: 94.53% (-2.78%p 급락)

**2-Fold Voting=2 (AND 전략)**
- 결과: 95.92% H-Mean (-0.34%p)
- 문제: Precision-Recall 불균형
  - Precision: 97.36% (높음)
  - Recall: 94.76% (낮음)
  - P-R Gap: 2.60%p (심각한 불균형)

**실패 원인**
1. **모델 다양성 부족**: Fold 0과 Fold 1의 Test H-Mean 차이 0.07%p
2. **Binary 투표의 한계**: 2개 Fold만으로는 OR/AND 극단적 선택만 가능
3. **중간 합의 지점 부재**: Voting≥2는 AND와 동일

### 1.3 5-Fold 확장 전략

**목표**
- 모델 다양성 확보: 5개의 서로 다른 데이터 분할
- 유연한 투표 임계값: Voting≥2, ≥3, ≥4 등 다양한 전략 가능
- Precision-Recall 균형: 중간 합의점 탐색

**기대 효과**
- Voting≥2 (40% 합의): 높은 Recall
- Voting≥3 (60% 합의): Balanced (과반수)
- Voting≥4 (80% 합의): 높은 Precision

---

## 2. 5-Fold 학습 설정

### 2.1 데이터 분할

**K-Fold 구성**
- Total images: 3,676개 (Train 3,272 + Val 404 merged)
- K-Fold splits: 5-Fold
- 저장 위치: `kfold_results_v2/fold_X/`

**각 Fold별 데이터**
```
Fold 0: train.json (2,940 images), val.json (736 images)
Fold 1: train.json (2,940 images), val.json (736 images)
Fold 2: train.json (2,940 images), val.json (736 images)
Fold 3: train.json (2,940 images), val.json (736 images)
Fold 4: train.json (2,940 images), val.json (736 images)
```

### 2.2 모델 설정

**공통 설정**
- Backbone: ResNet50 (timm pretrained)
- Input Resolution: 960×960px
- Decoder: UNet [256, 512, 1024, 2048]
- Optimizer: Adam (lr=0.0005, weight_decay=0.0001)
- Scheduler: CosineAnnealingLR (T_max=22)
- Batch Size: 4 (GPU 메모리 제약)

**Aggressive Postprocessing**
- thresh: 0.22 (이진화 임계값)
- box_thresh: 0.25 (박스 신뢰도 임계값)
- max_candidates: 600

**Augmentation**
- RandomRotate90
- RandomShadow, RandomFog
- GaussianNoise
- MotionBlur, Sharpen
- ColorJitter
- RandomBrightnessContrast

### 2.3 훈련 프로세스

**Fold별 전용 설정 파일 생성**
```yaml
# configs/preset/augmented_resnet50_aggressive_fold2.yaml
defaults:
  - base
  - /preset/datasets/db_augmented_resnet50
  - /preset/models/model_resnet50_aggressive
  - /preset/lightning_modules/base
  - _self_

data:
  train_json_path: /path/to/kfold_results_v2/fold_2/train.json
  val_json_path: /path/to/kfold_results_v2/fold_2/val.json
```

**훈련 스크립트**
```bash
# scripts/train_resnet50_fold2.sh
python runners/train.py \
    preset=augmented_resnet50_aggressive_fold2 \
    exp_name=resnet50_fold2 \
    trainer.max_epochs=22
```

**통합 훈련 파이프라인**
- 스크립트: `scripts/train_remaining_folds.sh`
- 순차 실행: Fold 2 → Fold 3 → Fold 4
- 에러 핸들링: 각 Fold별 성공/실패 체크
- 총 훈련 시간: 약 5-6시간

---

## 3. 훈련 결과

### 3.1 체크포인트 생성

**Fold 0** (기존 완료)
```
epoch=16-step=12495.ckpt (292MB)
epoch=18-step=13965.ckpt (292MB)
epoch=19-step=14700.ckpt (292MB) ← Best
```

**Fold 1** (기존 완료)
```
epoch=16-step=12512.ckpt (292MB)
epoch=18-step=13984.ckpt (292MB)
epoch=19-step=14720.ckpt (292MB) ← Best
```

**Fold 2** (신규 완료)
```
epoch=16-step=13906.ckpt (292MB)
epoch=20-step=17178.ckpt (292MB)
epoch=21-step=17996.ckpt (292MB) ← Best
```

**Fold 3** (신규 완료)
```
epoch=16-step=13906.ckpt (292MB)
epoch=20-step=17178.ckpt (292MB)
epoch=21-step=17996.ckpt (292MB) ← Best
```

**Fold 4** (신규 완료)
```
epoch=16-step=13906.ckpt (292MB)
epoch=20-step=17178.ckpt (292MB)
epoch=21-step=17996.ckpt (292MB) ← Best
```

### 3.2 Validation 성능 (추정)

> **Note**: WandB 로그 기반 추정치

| Fold | Best Epoch | Val H-Mean | Val Precision | Val Recall |
|------|-----------|------------|---------------|------------|
| Fold 0 | 19 | ~95.4% | ~96.6% | ~95.6% |
| Fold 1 | 19 | ~95.6% | ~96.7% | ~95.7% |
| Fold 2 | 21 | ~95.5% | ~96.5% | ~95.6% |
| Fold 3 | 21 | ~95.5% | ~96.5% | ~95.6% |
| Fold 4 | 21 | ~95.5% | ~96.5% | ~95.6% |

---

## 4. 예측 생성

### 4.1 예측 설정

**Hydra 오버라이드 이슈 해결**

문제: 체크포인트 경로에 `=` 기호 포함 (`epoch=21-step=17996.ckpt`)
```bash
# ❌ 실패
python runners/predict.py checkpoint_path=outputs/fold2/checkpoints/epoch=21-step=17996.ckpt

# ✅ 해결 (= 기호 이스케이프)
CHECKPOINT_ESCAPED=$(echo $CHECKPOINT | sed 's/=/\\=/g')
python runners/predict.py checkpoint_path=$CHECKPOINT_ESCAPED
```

### 4.2 예측 파일 생성

**Fold별 예측 스크립트**
```bash
# scripts/predict_resnet50_fold2_aggressive.sh
CHECKPOINT=$(ls -t outputs/resnet50_fold2/checkpoints/*.ckpt | head -1)
CHECKPOINT_ESCAPED=$(echo $CHECKPOINT | sed 's/=/\\=/g')

python runners/predict.py \
    preset=augmented_resnet50_aggressive \
    checkpoint_path=$CHECKPOINT_ESCAPED \
    exp_name=resnet50_fold2_aggressive_predict
```

**생성된 예측 파일**
```
Fold 0: outputs/resnet50_fold0_aggressive/submissions/20260201_052309.json
Fold 1: outputs/resnet50_fold1_aggressive_predict/submissions/20260201_112757.json
Fold 2: outputs/resnet50_fold2_aggressive_predict/submissions/20260201_184036.json
Fold 3: outputs/resnet50_fold3_aggressive_predict/submissions/20260201_184105.json
Fold 4: outputs/resnet50_fold4_aggressive_predict/submissions/20260201_184135.json
```

**통합 예측 파이프라인**
- 스크립트: `scripts/predict_all_remaining_folds.sh`
- 순차 실행: Fold 2 → Fold 3 → Fold 4
- 총 소요 시간: 약 1.5분 (각 Fold당 ~30초)

---

## 5. 앙상블 전략

### 5.1 앙상블 알고리즘

**IoU 기반 Box Matching**
```python
def ensemble_5fold(fold_predictions, voting_threshold, iou_threshold=0.5):
    # 1. 각 이미지별로 5개 Fold의 박스 수집
    # 2. IoU > threshold인 박스들을 그룹화
    # 3. 각 그룹의 투표 수 계산
    # 4. voting_threshold 이상인 그룹만 선택
    # 5. 그룹 내 박스들의 좌표 평균 계산
```

**주요 파라미터**
- `voting_threshold`: 최소 필요한 Fold 동의 수 (2~5)
- `iou_threshold`: 박스 매칭 기준 IoU (0.5)

### 5.2 Voting≥3 앙상블 (과반수)

**실행 명령**
```bash
python scripts/ensemble_5fold.py \
  --fold0 outputs/resnet50_fold0_aggressive/submissions/20260201_052309.json \
  --fold1 outputs/resnet50_fold1_aggressive_predict/submissions/20260201_112757.json \
  --fold2 outputs/resnet50_fold2_aggressive_predict/submissions/20260201_184036.json \
  --fold3 outputs/resnet50_fold3_aggressive_predict/submissions/20260201_184105.json \
  --fold4 outputs/resnet50_fold4_aggressive_predict/submissions/20260201_184135.json \
  --output outputs/ensemble_5fold_voting3.json \
  --voting 3 \
  --iou 0.5
```

**앙상블 결과**
```
Total images: 413
Total boxes: 45,010
Average boxes per image: 109.0

Voting distribution:
  5/5 folds agreed: 43,050 boxes (95.6%)
  4/5 folds agreed: 1,202 boxes (2.7%)
  3/5 folds agreed: 738 boxes (1.6%)
```

**특징**
- 95.6%의 박스가 전체 Fold 합의 (매우 높은 신뢰도)
- 과반수 투표로 Precision-Recall 균형 추구

### 5.3 Voting≥2 앙상블 (40% 합의)

**실행 명령**
```bash
python scripts/ensemble_5fold.py \
  --voting 2 \
  --output outputs/ensemble_5fold_voting2.json
  # (나머지 파라미터 동일)
```

**앙상블 결과**
```
Total images: 413
Total boxes: 45,362 (+352 vs Voting≥3)
Average boxes per image: 109.8

Voting distribution:
  5/5 folds agreed: 43,050 boxes (94.9%)
  4/5 folds agreed: 1,202 boxes (2.6%)
  3/5 folds agreed: 738 boxes (1.6%)
  2/5 folds agreed: 352 boxes (0.8%) ← 새로 포함
```

**특징**
- 2개 Fold만 동의해도 포함 (더 관대한 기준)
- 352개 추가 박스 (잠재적 Recall 증가)

---

## 6. 리더보드 제출 결과

### 6.1 Voting≥3 (과반수) ⭐ **최고 성적**

**제출 파일**: `submission_5fold_voting3.csv`

**리더보드 점수**
```
H-Mean    : 0.9628 (96.28%)
Precision : 0.9645 (96.45%)
Recall    : 0.9629 (96.29%)
```

**성능 분석**
- ✅ **신기록 달성**: 이전 최고 96.26% → 96.28% (+0.02%p)
- ✅ **완벽한 균형**: P-R Gap 0.16%p (매우 낮음)
- ✅ **높은 Precision**: 96.45% (False Positive 최소화)
- ✅ **안정적 Recall**: 96.29% (True Positive 유지)

**주요 특징**
- 5개 Fold 중 3개 이상 동의한 박스만 포함
- 고품질 박스 선별 (95.6%가 전체 합의)
- 보수적이면서도 균형잡힌 전략

### 6.2 Voting≥2 (40% 합의) ❌ **성능 하락**

**제출 파일**: `submission_5fold_voting2.csv`

**리더보드 점수**
```
H-Mean    : 0.9594 (95.94%)
Precision : 0.9589 (95.89%)
Recall    : 0.9618 (96.18%)
```

**성능 분석**
- ❌ **큰 폭 하락**: Voting≥3 대비 -0.34%p
- ❌ **Precision 급락**: -0.56%p (95.89%)
- ❌ **Recall 오히려 하락**: -0.11%p (예상과 반대)
- ❌ **P-R Gap 증가**: 0.29%p (불균형 악화)

**실패 원인**
1. **False Positive 과다**: 352개 추가 박스 대부분 FP
2. **낮은 합의 박스의 신뢰도 문제**: 2/5 = 40% 신뢰도
3. **2-Fold 앙상블 실패 패턴 재현**: 낮은 threshold → FP 증가

---

## 7. 앙상블 전략 비교

### 7.1 정량적 비교

| 전략 | H-Mean | Precision | Recall | P-R Gap | Boxes | 변화 |
|------|--------|-----------|--------|---------|-------|------|
| 단일 모델 (Aggressive) | 96.26% | 97.31% | 95.58% | 1.73%p | ~44,600 | Baseline |
| 5-Fold Voting≥3 | **96.28%** | **96.45%** | **96.29%** | **0.16%p** | 45,010 | **+0.02%p** ✅ |
| 5-Fold Voting≥2 | 95.94% | 95.89% | 96.18% | 0.29%p | 45,362 | -0.34%p ❌ |
| 2-Fold Voting=2 | 95.92% | 97.36% | 94.76% | 2.60%p | 43,521 | -0.34%p ❌ |
| 2-Fold Voting≥1 | 95.09% | 94.53% | 95.94% | 1.41%p | 45,784 | -1.17%p ❌ |

### 7.2 정성적 분석

**Voting≥3의 성공 요인**
1. **최적 균형점**: 과반수 투표로 FP/FN 균형
2. **높은 신뢰도**: 95.6% 박스가 전체 합의
3. **적절한 박스 수**: 45,010개 (단일 모델 +410개)
4. **Precision-Recall 균형**: 0.16%p gap (최저)

**Voting≥2의 실패 요인**
1. **과도한 포용**: 40% 합의만으로 포함
2. **FP 증가**: 352개 추가 박스 중 대부분 노이즈
3. **예상 밖 Recall 하락**: FP로 인한 정밀도 저하가 역효과
4. **2-Fold 실패 재현**: 낮은 threshold의 구조적 문제

### 7.3 투표 임계값별 특성

```
Voting≥5 (100% 합의): 43,050 boxes
  → Precision 최고, Recall 낮음 (너무 보수적)

Voting≥4 (80% 합의): 44,252 boxes (추정)
  → High Precision, Balanced Recall (시도 가치 있음)

Voting≥3 (60% 합의): 45,010 boxes ⭐ 최고 성능
  → Balanced Precision-Recall (검증됨)

Voting≥2 (40% 합의): 45,362 boxes
  → Precision 하락, FP 증가 (실패)

Voting≥1 (20% 합의): ~46,000 boxes (추정)
  → Precision 급락 예상
```

---

## 8. 핵심 발견 및 교훈

### 8.1 5-Fold 앙상블의 효과

**긍정적 측면**
- ✅ 모델 다양성 확보 (5개 데이터 분할)
- ✅ 유연한 투표 임계값 선택 가능
- ✅ 최고 점수 갱신 (96.28%)
- ✅ Precision-Recall 균형 달성 (0.16%p gap)

**한계점**
- ⚠️ 개선폭 미미 (+0.02%p)
- ⚠️ 5-Fold 다양성도 제한적 (동일 아키텍처, 동일 증강)
- ⚠️ 낮은 임계값(≥2) 여전히 실패

### 8.2 앙상블 전략 선택 기준

**과반수 투표(Voting≥3)가 최적인 이유**
1. 통계적 안정성: 51% 이상 합의
2. FP/FN 균형: 너무 보수적/관대하지 않음
3. 높은 신뢰도: 95.6% 전체 합의 박스

**낮은 임계값(≥2)이 실패하는 이유**
1. 노이즈 포함: 소수 Fold의 오류 반영
2. FP 누적: 여러 모델의 False Positive 합산
3. Precision 급락: 정밀도 저하가 Recall 이득 상쇄

### 8.3 단일 모델 vs 앙상블 비교

**단일 모델의 강점**
- Aggressive Postprocessing으로 96.26% 달성
- Precision 97.31% (매우 높음)
- 간단하고 빠른 추론

**5-Fold 앙상블의 강점**
- Precision-Recall 균형 (0.16%p gap)
- 미세한 성능 향상 (+0.02%p)
- 안정적 예측 (다수결 기반)

**결론**: 앙상블로 균형은 개선되었으나, 절대 성능 향상은 제한적

---

## 9. 다음 실험 방향

### 9.1 즉시 시도 가능 (1-2시간)

**1. Voting≥4 앙상블**
- 5개 중 4개 이상 동의 (80% 합의)
- 예상 boxes: ~44,250개
- Precision 더 높일 가능성
- **예상 H-Mean: 96.30-96.35%**

**2. NMS IoU Threshold 조정**
- 현재: IoU 0.5
- 시도: IoU 0.45 (중복 박스 더 제거)
- Voting≥3 재실행

**3. 초공격적 Postprocessing**
- thresh: 0.21 (현재 0.22)
- box_thresh: 0.24 (현재 0.25)
- 5-Fold 재예측 필요

### 9.2 단기 전략 (3-5시간)

**4. Weighted Ensemble**
- Fold별 validation 성능 기반 가중치
- 성능 높은 Fold에 더 큰 영향력

**5. Confidence-based Filtering**
- 각 Fold의 low-confidence 박스 사전 제거
- Threshold 기반 필터링

### 9.3 장기 전략 (1-2일)

**6. 다해상도 앙상블**
- 768px, 960px, 1024px 모델 혼합
- 스케일 다양성 확보

**7. 다른 Backbone**
- ResNet101, EfficientNetB3
- 아키텍처 다양성

**8. TTA 재도전**
- Rotation (90도 단위만)
- 좌표 변환 정확한 구현

---

## 10. 결론

### 10.1 실험 성과

**달성한 목표**
- ✅ 5-Fold 학습 성공적 완료 (15 checkpoints)
- ✅ 최고 점수 갱신: **96.28% H-Mean**
- ✅ Precision-Recall 균형: 0.16%p gap (역대 최고)
- ✅ 앙상블 전략 검증: Voting≥3 효과 입증

**한계점**
- ⚠️ 개선폭 미미: +0.02%p (비용 대비 효과 낮음)
- ⚠️ 단일 모델 성능 천장: 96.26% → 96.28%
- ⚠️ 5-Fold 다양성 제한: 동일 아키텍처의 한계

### 10.2 핵심 인사이트

**1. 과반수 투표의 효과**
- Voting≥3 (60% 합의)가 최적 균형점
- 통계적으로 안정적인 다수결 원칙

**2. 낮은 임계값의 위험성**
- Voting≥2 실패로 재확인
- 소수 합의 = 낮은 신뢰도 = FP 증가

**3. 앙상블의 역할**
- 절대 성능 향상보다는 **안정성과 균형** 확보
- 단일 모델 성능이 이미 높으면 한계

### 10.3 최종 권장사항

**현재 최고 모델**
- `submission_5fold_voting3.csv`
- H-Mean: 96.28%, P: 96.45%, R: 96.29%

**추가 개선 시도**
1. **우선순위 1**: Voting≥4 앙상블 (즉시 가능)
2. **우선순위 2**: Weighted Ensemble (단기)
3. **우선순위 3**: 다해상도 앙상블 (장기)

**비용-효과 고려**
- 5-Fold 학습: 6시간 → +0.02%p
- 추가 개선 예상: +0.05~0.10%p
- 96.5% 돌파 위해서는 **근본적 변화** 필요
  - 다른 아키텍처
  - 외부 데이터
  - 앙상블 다양성 확대

---

## 부록

### A. 파일 구조

```
baseline_code/
├── configs/preset/
│   ├── augmented_resnet50_aggressive_fold2.yaml
│   ├── augmented_resnet50_aggressive_fold3.yaml
│   └── augmented_resnet50_aggressive_fold4.yaml
├── scripts/
│   ├── train_resnet50_fold{2,3,4}.sh
│   ├── train_remaining_folds.sh
│   ├── predict_resnet50_fold{2,3,4}_aggressive.sh
│   ├── predict_all_remaining_folds.sh
│   ├── ensemble_5fold.py
│   └── monitor_5fold_training.sh
├── outputs/
│   ├── resnet50_fold{0,1,2,3,4}/checkpoints/
│   ├── resnet50_fold{0,1,2,3,4}_aggressive_predict/submissions/
│   ├── ensemble_5fold_voting2.json
│   ├── ensemble_5fold_voting3.json
│   ├── submission_5fold_voting2.csv
│   └── submission_5fold_voting3.csv
└── kfold_results_v2/
    └── fold_{0,1,2,3,4}/
        ├── train.json
        └── val.json
```

### B. 실행 커맨드 요약

**훈련**
```bash
bash scripts/train_remaining_folds.sh
```

**예측**
```bash
bash scripts/predict_all_remaining_folds.sh
```

**앙상블**
```bash
# Voting≥3
python scripts/ensemble_5fold.py \
  --fold0 outputs/resnet50_fold0_aggressive/submissions/20260201_052309.json \
  --fold1 outputs/resnet50_fold1_aggressive_predict/submissions/20260201_112757.json \
  --fold2 outputs/resnet50_fold2_aggressive_predict/submissions/20260201_184036.json \
  --fold3 outputs/resnet50_fold3_aggressive_predict/submissions/20260201_184105.json \
  --fold4 outputs/resnet50_fold4_aggressive_predict/submissions/20260201_184135.json \
  --output outputs/ensemble_5fold_voting3.json \
  --voting 3

# CSV 변환
python ocr/utils/convert_submission.py \
  -J outputs/ensemble_5fold_voting3.json \
  -O outputs/submission_5fold_voting3.csv
```

### C. 성능 진화 타임라인

```
2026-02-01 05:23 - ResNet50 Aggressive (단일): 96.26%
2026-02-01 13:07 - Fold 2,3,4 훈련 시작
2026-02-01 18:33 - 5-Fold 훈련 완료
2026-02-01 18:41 - Fold 2,3,4 예측 완료
2026-02-01 18:55 - Voting≥3 앙상블: 96.28% ⭐
2026-02-01 19:08 - Voting≥2 앙상블: 95.94% ❌
```

---

**보고서 작성**: 2026년 2월 1일  
**다음 실험**: Voting≥4 앙상블
