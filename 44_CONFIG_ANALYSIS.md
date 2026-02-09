# 📋 Config 구조 분석

## 현재 설정 (hrnet_w44_1024)

### 1. 메인 Config 계층구조
```
train.yaml (Main entry point)
  └─ pretrain_sroie_cord.yaml (or hrnet_w44_1024.yaml as preset)
     ├─ preset/base.yaml
     │  └─ Dataset/Model paths, Hydra config
     ├─ preset/datasets/db_augmented_1024.yaml
     │  └─ 1024×1024 이미지 전처리 & augmentation
     ├─ preset/models/model_hrnet_w44_hybrid_1024.yaml
     │  ├─ preset/models/encoder/timm_backbone_hrnet_w44.yaml
     │  ├─ preset/models/decoder/unet_hrnet_w44.yaml
     │  ├─ preset/models/head/db_head_lr_optimized.yaml
     │  └─ preset/models/loss/db_loss.yaml
     └─ preset/lightning_modules/base.yaml
```

---

## 📊 현재 파라미터 설정

### Data Configuration (db_augmented_1024.yaml)
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Image Path** | `/data/ephemeral/home/data/datasets/images/all` | 학습 이미지 경로 |
| **Annotation** | `train.json / val.json` | UFO JSON 형식 주석 |
| **Resolution** | 1024 × 1024 | LongestMaxSize + PadIfNeeded |
| **Batch Size** | 8 (Dataset Config에서) | 데이터 로더 배치 크기 |
| **Num Workers** | 4 | 병렬 데이터 로딩 |

### Augmentation Strategy
```yaml
Train Transform:
  - Geometric: Rotate (±12°), Perspective (0.05-0.1)
  - Color: RandomBrightnessContrast, HueSaturation, Equalize
  - Noise: GaussNoise, GaussianBlur
  - Normalization: ImageNet mean/std, ToTensorV2

Val/Test Transform:
  - Resize only (Longest=1024, Pad to 1024)
  - Normalization
```

### Training Configuration (model_hrnet_w44_hybrid_1024.yaml)
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Optimizer** | Adam | β₁=0.9, β₂=0.999 |
| **Learning Rate** | 0.001 | 고학습률 (외부 데이터 다양성 대응) |
| **Weight Decay** | 0.00008 | 온건한 정규화 |
| **Scheduler** | CosineAnnealingLR | T_max=30, η_min=0.00001 |
| **Max Epochs** | 40 | Early stopping 가능 (~epoch 20-30) |
| **Precision** | FP16 | 혼합 정밀도 (속도 향상) |

### Model Architecture
| 컴포넌트 | 설정 |
|---------|------|
| **Backbone** | HRNet-W44 (TIMM) |
| **Decoder** | UNet with HRNet-W44 features |
| **Head** | DB Head (Text Detection) |
| **Loss** | DB Loss (Dice + Binary Cross Entropy) |

---

## 🔄 Config 병합 순서

Hydra는 다음 순서로 병합:
1. **base.yaml** → 기본 paths, Hydra 설정
2. **db_augmented_1024.yaml** → 데이터셋 & 전처리
3. **model_hrnet_w44_hybrid_1024.yaml** → 모델, 옵티마이저, 스케줄러
4. **lightning_modules/base.yaml** → PyTorch Lightning 설정
5. **hrnet_w44_1024.yaml** (or pretrain_sroie_cord.yaml) → 최종 덮어쓰기

### 최종 병합된 Config 키:
```python
config = {
    'dataset_path': 'ocr.datasets',
    'model_path': 'ocr.models',
    'datasets': {
        'train_dataset': {...},
        'val_dataset': {...},
        ...
    },
    'transforms': {
        'train_transform': {...},
        'val_transform': {...},
        ...
    },
    'dataloader': {
        'batch_size': 8,  # Dataset config에서
        'num_workers': 4,
        ...
    },
    'models': {
        'optimizer': {...},
        'scheduler': {...},
        'encoder': {...},
        'decoder': {...},
        ...
    },
    'trainer': {
        'max_epochs': 40,
        'precision': 16,
        ...
    },
    'exp_name': 'hrnet_w44_1024_pretrain_stage1',
    ...
}
```

---

## ⚠️ 주의사항 & 개선 사항

### 현재 문제점
1. **Batch Size 불일치**
   - Dataset Config: batch_size=8
   - hrnet_w44_1024.yaml: batch_size=6 (덮어쓰기 가능)
   - **메모리 고려**: 1024×1024에서 batch=8은 높을 수 있음

2. **데이터 경로**
   - 고정: `/data/ephemeral/home/data/datasets/images/all`
   - 실제: `/data/ephemeral/home/data/datasets/images/` (all 폴더 없음)

3. **Augmentation**
   - `ToTensorV2` 호출 확인 필요
   - albumentations 버전 호환성 확인

### 권장 수정사항
```yaml
# 1. db_augmented_1024.yaml 수정
dataset_base_path: "/data/ephemeral/home/data/datasets/"
datasets:
  train_dataset:
    image_path: ${dataset_base_path}images  # /all 제거

# 2. hrnet_w44_1024.yaml 명시적 설정
dataloader:
  batch_size: 6  # 1024×1024에 더 안전
```

---

## 🚀 학습 실행 명령어

```bash
# Option 1: 기본 설정 사용
python runners/train.py

# Option 2: hrnet_w44_1024 preset 사용
python runners/train.py preset=hrnet_w44_1024

# Option 3: 커스텀 pretrain config
python runners/train.py --config-name=pretrain_sroie_cord

# Option 4: 파라미터 오버라이드
python runners/train.py \
    preset=hrnet_w44_1024 \
    trainer.max_epochs=50 \
    dataloader.batch_size=4
```

---

## 📈 기대 성능 & 시간

| 항목 | 값 |
|-----|-----|
| **데이터** | 3,272장 (대회) + SROIE/CORD (필요시) |
| **배치 크기** | 6 (1024×1024) |
| **에포크** | 40 (조기 종료 ~30) |
| **예상 시간** | 2-4일 (V100/A100 기준) |
| **예상 성능** | H-Mean 0.9880+ (Stage 1) |

