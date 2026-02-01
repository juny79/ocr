# ResNet50 성능 개선 전략 보고서

**현재 성능**: H-Mean 0.9620, Precision 0.9731, Recall 0.9535

---

## 📊 현재 상황 분석

| 지표 | 값 | 분석 |
|------|-----|------|
| **Precision** | 97.31% | ✅ 매우 높음 - 예측 박스는 거의 정확 |
| **Recall** | 95.35% | ⚠️ 개선 여지 - 약 4.65% 텍스트 미검출 |
| **H-Mean** | 96.20% | 🎯 목표: 96.5~97.0% |

**핵심 문제**: Precision이 Recall보다 높음 → 모델이 **보수적으로 예측**

**개선 방향**: Recall을 높이되 Precision 손실 최소화

---

## 🚀 즉시 실행 가능한 개선 전략 (우선순위 순)

### 1️⃣ 후처리 임계값 최적화 ⭐⭐⭐⭐⭐
**예상 개선**: +0.3~0.5% | **소요 시간**: 30초 | **난이도**: ⭐

**변경 사항**:
```yaml
# 기존 (현재 제출)
thresh: 0.25
box_thresh: 0.3
max_candidates: 500

# 공격적 설정 (방금 생성)
thresh: 0.22        # 이진화 임계값 낮춤 → 더 많은 영역 감지
box_thresh: 0.25    # 박스 신뢰도 임계값 낮춤 → 더 많은 박스 허용
max_candidates: 600 # 최대 박스 수 증가
```

**실행**:
```bash
# 이미 생성 완료!
# 파일: outputs/submission_resnet50_aggressive.csv
```

**예상 결과**:
- Recall: 95.35% → **96.0~96.5%** (+0.65~1.15%)
- Precision: 97.31% → 96.8~97.0% (-0.31~0.51% 감소)
- H-Mean: 96.20% → **96.4~96.8%** (+0.2~0.6%)

---

### 2️⃣ K-Fold 앙상블 (Fold 0 + 다른 Fold) ⭐⭐⭐⭐
**예상 개선**: +0.5~1.0% | **소요 시간**: 4시간 (Fold 1-2 추가 훈련) | **난이도**: ⭐⭐

**전략**:
- Fold 0 이미 완료 (H-Mean 95.89%)
- Fold 1, 2 추가 훈련 (각 2시간)
- 3-Fold Voting 앙상블

**실행**:
```bash
# Fold 1 훈련
cd /data/ephemeral/home/baseline_code
python runners/train.py \
    preset=augmented_resnet50 \
    ++datasets.train_dataset.annotation_path=kfold_results_v2/fold_1/train.json \
    ++datasets.val_dataset.annotation_path=kfold_results_v2/fold_1/val.json \
    ++trainer.max_epochs=22 \
    exp_name="resnet50_fold1" \
    wandb=True

# Fold 2 훈련 (동일 방식)

# 앙상블 예측
python scripts/ensemble_kfold.py --folds 0 1 2 --strategy voting --threshold 2
```

**예상 결과**:
- H-Mean: 96.20% → **96.7~97.2%**
- 다양성 확보로 오류 보완

---

### 3️⃣ Test-Time Augmentation (TTA) ⭐⭐⭐⭐
**예상 개선**: +0.2~0.4% | **소요 시간**: 5분 | **난이도**: ⭐⭐

**전략**:
- 원본 + 수평 플립 예측
- 두 예측 결과 병합

**실행**:
```bash
# TTA 스크립트 실행
cd /data/ephemeral/home/baseline_code
python scripts/predict_with_tta.py \
    --checkpoint outputs/resnet50_fold0/checkpoints/epoch=19-step=14700.ckpt \
    --preset augmented_resnet50_aggressive \
    --output outputs/tta_predictions

# CSV 변환
python ocr/utils/convert_submission.py \
    -J outputs/tta_predictions/tta_predictions.json \
    -O outputs/submission_resnet50_tta.csv
```

**예상 결과**:
- 경계선 부근 불확실성 감소
- H-Mean: 96.20% → **96.4~96.6%**

---

### 4️⃣ 후처리 그리드 서치 ⭐⭐⭐
**예상 개선**: +0.3~0.6% | **소요 시간**: 10분 | **난이도**: ⭐⭐

**전략**:
- 여러 임계값 조합 테스트
- Validation set에서 최적 조합 찾기

**실행 스크립트**:
```python
# scripts/grid_search_postprocess.py
import itertools
from tqdm import tqdm

# 테스트할 값들
thresh_values = [0.20, 0.22, 0.25, 0.28]
box_thresh_values = [0.23, 0.25, 0.28, 0.30]
max_candidates_values = [500, 600, 700]

best_hmean = 0
best_config = {}

for thresh, box_thresh, max_cand in itertools.product(
    thresh_values, box_thresh_values, max_candidates_values
):
    # 예측 실행 (Validation set)
    hmean = evaluate_with_config(thresh, box_thresh, max_cand)
    
    if hmean > best_hmean:
        best_hmean = hmean
        best_config = {
            'thresh': thresh,
            'box_thresh': box_thresh,
            'max_candidates': max_cand
        }

print(f"Best config: {best_config}")
print(f"Best H-Mean: {best_hmean}")
```

---

### 5️⃣ 배치 크기 증가 재훈련 ⭐⭐⭐
**예상 개선**: +0.2~0.3% | **소요 시간**: 2시간 | **난이도**: ⭐⭐⭐

**전략**:
- Mixed Precision (FP16) 활성화
- 배치 크기 4 → 8로 증가
- 더 안정적인 그래디언트

**설정 변경**:
```yaml
# configs/preset/datasets/db_augmented_resnet50.yaml
dataloaders:
  train_dataloader:
    batch_size: 8  # 4 → 8

# runners/train.py
trainer:
  precision: 16  # FP32 → FP16
  amp_backend: 'native'
```

**실행**:
```bash
python runners/train.py \
    preset=augmented_resnet50 \
    ++datasets.train_dataset.annotation_path=kfold_results_v2/fold_0/train.json \
    ++datasets.val_dataset.annotation_path=kfold_results_v2/fold_0/val.json \
    ++trainer.max_epochs=22 \
    ++trainer.precision=16 \
    exp_name="resnet50_fold0_fp16_bs8" \
    wandb=True
```

---

## 📋 실행 순서 추천

### Phase 1: 즉시 실행 (10분 이내)
1. ✅ **submission_resnet50_aggressive.csv 제출** (이미 생성 완료)
   - 예상: H-Mean 96.4~96.6%
   
2. **TTA 예측 실행 및 제출** (5분)
   ```bash
   cd /data/ephemeral/home/baseline_code
   python scripts/predict_with_tta.py \
       --checkpoint outputs/resnet50_fold0/checkpoints/epoch=19-step=14700.ckpt
   ```

### Phase 2: 단기 개선 (1시간 이내)
3. **후처리 그리드 서치** (10분)
   - Validation set에서 최적 임계값 찾기
   
4. **최적 임계값으로 재예측** (1분)

### Phase 3: 중기 개선 (4시간)
5. **Fold 1, 2 추가 훈련** (각 2시간)
6. **3-Fold 앙상블** (5분)
   - 예상: H-Mean 96.8~97.2%

---

## 🎯 예상 최종 성능

| 전략 | H-Mean | 누적 개선 |
|------|--------|---------|
| **현재** | 96.20% | - |
| + Aggressive Postprocess | 96.50% | +0.30% |
| + TTA | 96.70% | +0.50% |
| + 3-Fold Ensemble | **97.10%** | **+0.90%** |

---

## 📂 생성된 파일

### 설정 파일
- `configs/preset/models/head/db_head_aggressive.yaml` - 공격적 후처리
- `configs/preset/models/model_resnet50_aggressive.yaml` - 공격적 모델
- `configs/preset/augmented_resnet50_aggressive.yaml` - 공격적 프리셋
- `configs/predict_resnet50_aggressive.yaml` - 예측 설정

### 실행 스크립트
- `scripts/predict_with_tta.py` - TTA 예측 스크립트

### 제출 파일
- ✅ `outputs/submission_resnet50_aggressive.csv` - 공격적 후처리 (즉시 제출 가능)

---

## 🔬 추가 실험 아이디어 (장기)

### 6️⃣ 더 큰 백본 (EfficientNet-B3/B4)
- ResNet50 → EfficientNet-B3
- 예상 개선: +0.5~0.8%
- 소요 시간: 3시간

### 7️⃣ Pseudo-Labeling
- Test set을 고신뢰도 예측으로 라벨링
- 재훈련으로 일반화 향상
- 예상 개선: +0.3~0.5%

### 8️⃣ 외부 데이터 활용
- CORD, SROIE 등 영수증 데이터셋
- 사전 훈련 후 Fine-tuning
- 예상 개선: +0.5~1.0%

---

## 💡 핵심 인사이트

1. **Recall 향상이 우선**: 현재 Precision이 충분히 높으므로 Recall 올리기
2. **후처리가 가장 빠름**: 재훈련 없이 즉시 개선 가능
3. **앙상블이 가장 강력**: K-Fold로 안정적으로 1% 향상 가능
4. **시간 vs 성능 트레이드오프**: 
   - 30초: +0.3% (후처리)
   - 4시간: +0.9% (앙상블)

---

**권장 행동**:
1. 먼저 **submission_resnet50_aggressive.csv** 제출하여 효과 확인
2. 효과 있으면 TTA 추가 적용
3. 시간 여유 있으면 Fold 1-2 훈련하여 앙상블

**최종 목표**: H-Mean **97.0%+** 달성 가능! 🎯
