# EfficientNet-B4 Learning Rate 최적화 전략
## 현재 → 목표: 96.37% → 96.50%+

---

## 📊 현재 상황

### Postprocessing 최적화 결과 (완료 ✅)

**제출 결과 (4개)**:
| Rank | Config | H-Mean | Precision | Recall | P-R Gap |
|------|--------|--------|-----------|--------|---------|
| 🥇 1 | t0.28_b0.25 | **96.37%** | 96.74% | 96.23% | 0.51%p |
| 🥈 2 | t0.27_b0.26 | 96.29% | 96.70% | 96.14% | 0.56%p |
| 🥉 3 | t0.26_b0.28 | 96.14% | 96.78% | 95.80% | 0.98%p |
| 4 | t0.25_b0.27 | 96.06% | 96.56% | 95.85% | 0.71%p |

**원본 (Base)**:
- H-Mean: 96.00%
- Precision: 96.27%
- Recall: 95.98%
- Config: thresh=0.22, box_thresh=0.25

### 성능 개선

✅ **H-Mean**: 96.00% → **96.37%** (+0.37%p)  
✅ **Precision**: 96.27% → **96.74%** (+0.47%p)  
✅ **Recall**: 95.98% → **96.23%** (+0.25%p)

### ResNet50 비교

| Metric | ResNet50 5-Fold | EfficientNet-B4 Optimized | 차이 |
|--------|-----------------|---------------------------|------|
| H-Mean | 96.28% | **96.37%** | **+0.09%p** ✅ |
| Precision | 97.31% | 96.74% | -0.57%p |
| Recall | 95.58% | 96.23% | **+0.65%p** ✅ |

---

## 🎯 핵심 인사이트

### 1. ResNet50 능가 달성! 🎉
- EfficientNet-B4가 **단일 모델**로 ResNet50 5-Fold 앙상블을 넘어섬
- Postprocessing 최적화만으로 달성

### 2. Recall이 핵심 개선 포인트
- Recall: +0.65%p 대폭 개선
- EfficientNet-B4가 더 많은 박스를 정확하게 검출

### 3. 최적 Postprocessing 파라미터
- **thresh = 0.28** (예상 0.25-0.26보다 높음)
- **box_thresh = 0.25** (낮은 값 유지)
- 이유: 기존 모델이 False Positive가 많았음 (thresh 증가로 해결)

### 4. Trade-off 분석
```
thresh 효과 (0.22 → 0.28):
  - Precision: +0.47%p ✅
  - Recall: +0.25%p ✅ (예상과 반대!)
  → False Positive가 많아서 thresh 증가가 양쪽 개선

box_thresh 효과:
  - 0.25 vs 0.26: 큰 차이 없음
  - 0.28로 증가: Recall -0.33%p (과도)
  → 낮게 유지하는 것이 유리
```

### 5. P-R Balance
- **t0.28_b0.25**: 0.51%p (최적 균형) ✅
- t0.27_b0.26: 0.56%p
- t0.26_b0.28: 0.98%p (불균형)

---

## 🚀 다음 전략: Learning Rate 최적화

### 목표
- 현재: 96.37% (thresh=0.28, LR=0.0003)
- 목표: **96.50%+** (LR 최적화)
- 최종: **96.70-96.80%** (5-Fold Ensemble)

### 가설
현재 Learning Rate (0.0003)가 EfficientNet-B4에 최적이 아닐 수 있음:
- ResNet50: LR=0.0005 최적
- EfficientNet-B4: LR=0.0003 (60% 감소)
- 하지만 실제로는 0.0004-0.0005가 더 나을 수도?

### WandB Sweep 전략

**방법**: Bayesian Optimization  
**실행 횟수**: 12회 (LR에 집중)  
**소요 시간**: 약 4시간

**탐색 파라미터**:
```yaml
Critical (High Impact):
  - Learning Rate: 0.00025 - 0.0006
  - Weight Decay: 0.00005 - 0.0005

Secondary (Medium Impact):
  - T_Max: 20, 22, 24
  - eta_min: 0.000005 - 0.00005

Fixed (최적값):
  - thresh: 0.28
  - box_thresh: 0.25
  - max_candidates: 600
```

**예상 결과**:
- Conservative (70%): 96.42-96.48%
- Neutral (50%): 96.48-96.55%
- Optimistic (30%): 96.55-96.65%

---

## 📋 실행 계획

### Option 1: WandB Sweep (권장)

**장점**:
- ✅ 자동 최적화
- ✅ Bayesian으로 효율적 탐색
- ✅ 96.50%+ 달성 가능성 높음

**실행**:
```bash
cd /data/ephemeral/home/baseline_code

# WandB 로그인
wandb login

# Sweep 실행 (12회, 4시간)
bash scripts/run_sweep_lr_optimized.sh 12

# 또는 Background 실행
bash scripts/run_sweep_lr_optimized.sh 12 bg
```

**다음 단계**:
1. WandB 대시보드에서 최고 성능 확인
2. 최적 LR로 단일 모델 재학습
3. 96.50%+ 달성 시 5-Fold 진행

---

### Option 2: 수동 LR 테스트 (빠름)

**장점**:
- ✅ WandB 불필요
- ✅ 2-3시간 소요
- ✅ 단순 명확

**실행**:
```bash
# LR=0.0004 테스트 (현재 0.0003의 133%)
python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_lr_0.0004 \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22

# 결과 확인 후 예측
python runners/predict.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_lr_0.0004_predict \
  checkpoint_path=outputs/efficientnet_b4_lr_0.0004/checkpoints/best.ckpt
```

**LR 후보**:
1. **0.0004** (현재 0.0003의 133%) - 추천 ⭐
2. 0.00035 (현재 0.0003의 117%)
3. 0.00045 (현재 0.0003의 150%)

---

### Option 3: 바로 5-Fold (보수적)

**장점**:
- ✅ 즉시 실행 가능
- ✅ 96.45-96.55% 기대
- ✅ 안정적

**실행**:
```bash
# 현재 최적 설정(thresh=0.28, LR=0.0003)으로 5-Fold
bash scripts/train_efficientnet_b4_5fold_optimized.sh
```

**예상 결과**:
- Conservative: 96.42-96.48%
- Neutral: 96.48-96.55%
- Optimistic: 96.55-96.62%

---

## 🎯 권장 실행 순서

### Phase 1: 수동 LR 테스트 (2-3시간)
```bash
# LR=0.0004 테스트
python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_lr_0.0004 \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22
```

### Phase 2: 결과 평가 (30분)
- 96.45%+ 달성 → Phase 3 (5-Fold)
- 96.40-96.45% → WandB Sweep
- <96.40% → LR=0.00035 재시도

### Phase 3: 5-Fold 학습 (12시간)
```bash
# 최적 LR로 5-Fold
bash scripts/train_efficientnet_b4_5fold_optimized.sh
```

### Phase 4: Voting Ensemble
- Voting≥3 추천
- 예상: 96.55-96.65%

---

## 📊 예상 최종 성능

### 단일 모델 (LR 최적화)
- Conservative: 96.45-96.48%
- Neutral: 96.48-96.55%
- Optimistic: 96.55-96.60%

### 5-Fold Ensemble (Voting≥3)
- Conservative: 96.50-96.58%
- Neutral: 96.58-96.65%
- Optimistic: 96.65-96.75%

### vs ResNet50
| Model | H-Mean | 차이 |
|-------|--------|------|
| ResNet50 5-Fold | 96.28% | - |
| EfficientNet-B4 Single (Optimized) | 96.50% | +0.22%p |
| EfficientNet-B4 5-Fold (Voting≥3) | 96.65% | +0.37%p |

---

## 🔧 생성된 파일

### Config Files
1. **configs/sweep_efficientnet_b4_lr_optimized.yaml**
   - WandB Sweep 설정
   - Postprocessing 고정 (thresh=0.28, box_thresh=0.25)
   - LR, Weight Decay 탐색

2. **configs/preset/efficientnet_b4_lr_optimized.yaml**
   - 최적화된 preset

3. **configs/preset/models/model_efficientnet_b4_lr_optimized.yaml**
   - LR=0.0004 (기본값, Sweep으로 최적화)

4. **configs/preset/models/head/db_head_lr_optimized.yaml**
   - thresh=0.28, box_thresh=0.25 고정

### Scripts
5. **scripts/run_sweep_lr_optimized.sh**
   - WandB Sweep 실행 스크립트
   - Background 모드 지원

---

## 💡 핵심 요약

### 달성한 것
✅ Postprocessing 최적화로 96.37% 달성  
✅ ResNet50 5-Fold (96.28%) 능가  
✅ 최적 파라미터 발견: thresh=0.28, box_thresh=0.25

### 다음 목표
🎯 Learning Rate 최적화로 96.50%+ 달성  
🎯 5-Fold Ensemble로 96.65%+ 달성  
🎯 최종 목표: 96.70-96.80%

### 추천 전략
1️⃣ **즉시**: LR=0.0004 단일 테스트 (2시간)  
2️⃣ **96.45%+ 달성 시**: 5-Fold 진행 (12시간)  
3️⃣ **<96.45% 시**: WandB Sweep (4시간)

---

## 🚀 바로 시작하기

**가장 빠른 방법** (2시간):
```bash
cd /data/ephemeral/home/baseline_code
python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_lr_0.0004 \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22
```

**가장 확실한 방법** (4시간):
```bash
wandb login
bash scripts/run_sweep_lr_optimized.sh 12 bg
```

**가장 안전한 방법** (12시간):
```bash
bash scripts/train_efficientnet_b4_5fold_optimized.sh
```

---

**생성 일시**: 2026-02-02  
**현재 Best**: 96.37% (EfficientNet-B4, thresh=0.28)  
**목표**: 96.65%+ (5-Fold Ensemble)  
**전략**: LR 최적화 → 5-Fold → 96.7% 돌파!
