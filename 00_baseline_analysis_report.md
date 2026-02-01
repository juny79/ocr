# Baseline 코드 종합 분석 보고서

## 📊 1. 현재 성능 지표 (10 Epochs)

```
H-Mean (F1-Score): 0.8818
Precision: 0.9651  ⬆️ (매우 높음)
Recall: 0.8194     ⬇️ (개선 필요)
```

### 핵심 문제 진단

- **높은 Precision (96.51%)**: 모델이 예측한 박스는 대부분 정확함
- **낮은 Recall (81.94%)**: 많은 텍스트 영역을 놓치고 있음 (약 18% 미탐지)
- **불균형 패턴**: 모델이 **보수적으로 예측**하여 확실한 것만 감지

---

## 🏗️ 2. 모델 아키텍처 분석

### 2.1 전체 구조 (DBNet 기반)

```
Input Image (640×640)
    ↓
Encoder: ResNet18 (Pretrained)
    ↓
Decoder: UNet (FPN-style)
    ↓
Head: DBHead (Prob + Thresh + Binary Maps)
    ↓
Loss: DBLoss (BCE + L1 + Dice)
```

### 2.2 Encoder: TimmBackbone (ResNet18)

```yaml
model_name: 'resnet18'
pretrained: true
select_features: [1, 2, 3, 4]  # Multi-scale features
```

**특징:**
- ✅ **ResNet18**: 가벼운 백본 (11.7M params)
- ⚠️ **한계**: 작은 텍스트나 복잡한 레이아웃 감지 능력 제한
- 💡 **제안**: ResNet34/50, EfficientNet, ConvNeXt 등으로 업그레이드 가능

### 2.3 Decoder: UNet

```yaml
in_channels: [64, 128, 256, 512]
inner_channels: 256
output_channels: 64
strides: [4, 8, 16, 32]
```

**특징:**
- ✅ Multi-scale feature fusion
- ⚠️ `inner_channels: 256`은 중간 수준 (더 증가 가능)

### 2.4 Head: DBHead

```yaml
in_channels: 256
upscale: 4
k: 50  # Differentiable Binarization 계수
postprocess:
  thresh: 0.3          # 🔴 중요
  box_thresh: 0.4      # 🔴 중요
  max_candidates: 300
```

**Recall 저하의 주요 원인:**
- ⚠️ **`box_thresh: 0.4`**: 너무 높음 → 확신이 높은 박스만 선택
- ⚠️ **`thresh: 0.3`**: 이진화 임계값
- 💡 **Recall을 높이려면 `box_thresh`를 낮춰야 함** (0.25~0.35)

---

## 📐 3. 데이터 전처리 분석

### 3.1 이미지 크기

```yaml
transforms:
  - LongestMaxSize: 640
  - PadIfNeeded: 640×640
```

**평가:**
- ⚠️ **640×640은 작음**: Receipt 이미지는 종종 세로로 길고 텍스트가 작음
- 💡 **제안**: 800~1024로 증가 (메모리 허용 시)

### 3.2 Data Augmentation

```yaml
train_transform:
  - HorizontalFlip: p=0.5
  - Normalize: ImageNet 기준
```

**문제점:**
- ❌ **증강이 매우 부족**
  - Rotation ❌
  - RandomBrightnessContrast ❌
  - ColorJitter ❌
  - RandomScale ❌
  - ShiftScaleRotate ❌
  
**영향:**
- 다양한 각도/조명/스케일의 텍스트 일반화 부족
- **Recall 저하**에 기여

### 3.3 Collate Function

```yaml
shrink_ratio: 0.4    # Text shrinkage for probability map
thresh_min: 0.3
thresh_max: 0.7
```

**특징:**
- ✅ DBNet 표준 파라미터 사용
- ⚠️ `shrink_ratio: 0.4`는 중간값 (0.3~0.5 범위 실험 가능)

---

## 🔥 4. 손실 함수 분석

```yaml
DBLoss:
  negative_ratio: 3.0              # Hard negative mining
  prob_map_loss_weight: 5.0        # 🔴
  thresh_map_loss_weight: 10.0     # 🔴
  binary_map_loss_weight: 1.0      # 🔴
```

**구성:**
- **Probability Map Loss**: BCE Loss (binary text/non-text)
- **Threshold Map Loss**: L1 Loss (adaptive threshold)
- **Binary Map Loss**: Dice Loss (differentiable binarization)

**가중치 비율:**
```
Prob : Thresh : Binary = 5 : 10 : 1
```

**평가:**
- ✅ **Threshold map에 높은 가중치**: DBNet 논문과 일치
- ⚠️ **Binary map 가중치가 낮음**: 최종 검출 성능에 직접적 영향
- 💡 **제안**: `binary_map_loss_weight: 2.0~3.0`으로 증가 실험

---

## ⚙️ 5. 학습 설정 분석

### 5.1 Optimizer & Scheduler

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001                    # 🔴 Learning rate
  weight_decay: 0.0001

scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 100               # ⚠️ 너무 큼
  gamma: 0.1
```

**문제점:**
- ⚠️ **`step_size: 100`**: 10 epoch 학습에서는 작동 안 함
  - Step이 한 번도 발동되지 않음
- ⚠️ **StepLR**: 구식 스케줄러
- 💡 **제안**: 
  - `CosineAnnealingLR`, `ReduceLROnPlateau` 사용
  - `step_size: 3~5`로 조정

### 5.2 Training Config

```yaml
trainer:
  max_epochs: 10                # 🔴 너무 짧음
  batch_size: 16
  num_workers: 4
```

**평가:**
- ⚠️ **10 epochs**: DBNet은 일반적으로 300~1200 epochs 필요
- ⚠️ **batch_size: 16**: 메모리 여유 있으면 증가 가능
- 💡 **제안**: 최소 50~100 epochs

---

## 🎯 6. Postprocessing 파라미터 영향 분석

### 현재 설정의 동작 원리

```python
# db_postprocess.py
thresh: 0.3          # Probability map을 0.3 기준으로 이진화
box_thresh: 0.4      # 박스 신뢰도 0.4 이상만 채택
max_candidates: 300  # 최대 300개 후보
```

**Recall 저하 시나리오:**
1. 모델이 작은/회전된 텍스트에 대해 0.35 신뢰도로 예측
2. `box_thresh: 0.4` 기준 미달로 **제거됨**
3. 결과: False Negative 증가 → **Recall 하락**

**개선 방향:**
```yaml
postprocess:
  thresh: 0.25           # 0.3 → 0.25 (더 민감하게)
  box_thresh: 0.3        # 0.4 → 0.3 (임계값 완화)
  max_candidates: 500    # 300 → 500 (더 많은 후보)
```

---

## 📈 7. 성능 병목 요인 우선순위

### 🔴 Critical (즉시 개선 필요)

1. **`box_thresh: 0.4 → 0.3`**: Recall 즉시 상승 예상
2. **Scheduler 수정**: `step_size: 100 → 3~5`
3. **Epochs 증가**: `10 → 50+`

### 🟡 High Impact (중요)

4. **Data Augmentation 추가**: 
   - Rotation, Brightness, Scale 변환
5. **이미지 해상도 증가**: `640 → 800`
6. **Binary map loss 가중치**: `1.0 → 2.0`

### 🟢 Medium Impact (실험 가치)

7. **Backbone 업그레이드**: ResNet18 → ResNet34/50
8. **Optimizer 변경**: Adam → AdamW
9. **Shrink ratio 조정**: 0.4 → 0.3

---

## 💡 8. 즉시 적용 가능한 Quick Wins

### Phase 1: Postprocessing 조정 (재학습 불필요)

**파일**: `configs/preset/models/head/db_head.yaml`

```yaml
postprocess:
  thresh: 0.25           # ⬇️ 0.3 → 0.25
  box_thresh: 0.3        # ⬇️ 0.4 → 0.3
  max_candidates: 500    # ⬆️ 300 → 500
```

**예상 효과:** Recall +2~5%, Precision -1~2%, H-Mean +1~3%

### Phase 2: Scheduler 수정

**파일**: `configs/preset/models/model_example.yaml`

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 10              # max_epochs와 동일
  eta_min: 0.00001
```

### Phase 3: Augmentation 추가

**파일**: `configs/preset/datasets/db.yaml`

```yaml
train_transform:
  transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
    # 🆕 추가
    - _target_: albumentations.Rotate
      limit: 10
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    - _target_: albumentations.GaussNoise
      p: 0.3
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

---

## 📊 9. 예상 성능 개선 로드맵

| 단계 | 조치 | 예상 Recall | 예상 Precision | 예상 H-Mean |
|------|------|-------------|----------------|-------------|
| **Baseline** | - | **0.8194** | **0.9651** | **0.8818** |
| **Step 1** | box_thresh 조정 | 0.845 (+3%) | 0.950 (-1.5%) | 0.895 (+1.3%) |
| **Step 2** | Augmentation 추가 | 0.860 (+5%) | 0.945 (-2%) | 0.901 (+2%) |
| **Step 3** | Epochs 50 + Scheduler | 0.880 (+7.4%) | 0.940 (-2.6%) | 0.909 (+3%) |
| **Step 4** | Image size 800 | 0.895 (+9.2%) | 0.935 (-3.1%) | 0.915 (+3.7%) |
| **Step 5** | Backbone 업그레이드 | 0.910 (+11%) | 0.930 (-3.6%) | 0.920 (+4.2%) |

---

## 🎓 10. 결론 및 권장사항

### 강점

✅ DBNet 아키텍처 구현 완성도 높음  
✅ Hydra 기반 설정 관리로 실험 용이  
✅ CLEval 평가 체계 잘 구축됨  
✅ 높은 Precision → False Positive 적음  

### 약점

❌ **Recall 부족** (18% 미탐지)  
❌ Data Augmentation 거의 없음  
❌ Scheduler가 10 epoch에서 작동 안 함  
❌ Postprocessing threshold가 너무 보수적  
❌ 학습 epochs 부족 (10 vs 권장 300+)  

### 최우선 개선 항목

1. **`box_thresh: 0.4 → 0.3`** (5분 작업, 즉시 효과)
2. **Scheduler 수정** (CosineAnnealing으로 변경)
3. **Augmentation 추가** (Rotate, Brightness, Scale)
4. **Epochs 증가** (50~100 epochs)

### 장기 개선 방향

- Backbone 업그레이드 (ResNet34/50, EfficientNet)
- Pseudo labeling 활용 (data/pseudo_label 활용)
- Multi-scale training/inference
- Ensemble 전략

---

## 📝 상세 코드 분석

### 주요 파일 경로

```
baseline_code/
├── configs/
│   ├── train.yaml                          # 학습 설정
│   ├── test.yaml                           # 테스트 설정
│   ├── predict.yaml                        # 예측 설정
│   └── preset/
│       ├── datasets/db.yaml                # 데이터셋 & Transform
│       ├── models/
│       │   ├── model_example.yaml          # Optimizer & Scheduler
│       │   ├── encoder/timm_backbone.yaml  # ResNet18
│       │   ├── decoder/unet.yaml           # UNet Decoder
│       │   ├── head/db_head.yaml           # DBHead & Postprocess
│       │   └── loss/db_loss.yaml           # DBLoss
│       └── lightning_modules/base.yaml     # Lightning 설정
├── ocr/
│   ├── datasets/
│   │   ├── base.py                         # OCRDataset
│   │   ├── db_collate_fn.py                # Ground truth map 생성
│   │   └── transforms.py                   # Albumentations wrapper
│   ├── models/
│   │   ├── architecture.py                 # OCRModel (전체 파이프라인)
│   │   ├── encoder/timm_backbone.py        # TimmBackbone
│   │   ├── decoder/unet.py                 # UNet
│   │   ├── head/
│   │   │   ├── db_head.py                  # DBHead
│   │   │   └── db_postprocess.py           # 후처리 (박스 추출)
│   │   └── loss/
│   │       ├── db_loss.py                  # DBLoss
│   │       ├── bce_loss.py                 # BCE Loss
│   │       ├── dice_loss.py                # Dice Loss
│   │       └── l1_loss.py                  # L1 Loss
│   ├── lightning_modules/
│   │   └── ocr_pl.py                       # Lightning Module
│   └── metrics/
│       └── cleval_metric.py                # CLEval 평가
└── runners/
    ├── train.py                            # 학습 실행
    ├── test.py                             # 테스트 실행
    └── predict.py                          # 예측 실행
```

---

## 🔬 추가 실험 아이디어

### 1. Multi-scale Training
- 다양한 이미지 크기로 학습 (640, 800, 1024)
- Validation은 고정 크기로 평가

### 2. Test Time Augmentation (TTA)
- Horizontal flip
- Multi-scale inference
- 결과 앙상블

### 3. Pseudo Labeling 활용
```
data/pseudo_label/
├── cord-v2/
├── sroie/
└── wildreceipt/
```
- 외부 데이터셋 활용하여 사전 학습
- Fine-tuning on target dataset

### 4. Loss Function 실험
- Focal Loss 추가
- IoU Loss 추가
- Weighted combination

### 5. Backbone Ablation Study
- ResNet34, ResNet50
- EfficientNet-B0, B1, B2
- ConvNeXt-Tiny
- MobileNetV3

---

## 📚 참고 자료

### 논문
- **DBNet**: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
- **CLEval**: [Character-Level Evaluation for Text Detection](https://github.com/clovaai/CLEval)

### 구현 참조
- [MhLiao/DB](https://github.com/MhLiao/DB/) - 공식 DBNet 구현

---

**보고서 작성일**: 2026년 1월 29일  
**분석 대상**: baseline_code (10 epochs 학습 결과)  
**현재 성능**: H-Mean 0.8818, Precision 0.9651, Recall 0.8194  

---

**현재 코드는 견고한 기반을 갖추었으나, Recall 개선에 집중한 파라미터 튜닝과 Data Augmentation이 급선무입니다.**
