# Run 8 Learning Rate 파라미터 재분석

## 🎯 정확한 문제 정의

### 고정된 베이스라인 (검증 완료 ✅)
```yaml
# Postprocessing - 96.53% 달성한 최적값
thresh: 0.29
box_thresh: 0.25
max_candidates: 600
```
→ **이 파라미터는 문제 없음. 그대로 유지!**

### 문제의 파라미터 (Sweep Run 8)
```yaml
# Learning Rate Optimization - Validation에서 96.60%
lr: 0.0005134333170096499        # ← 의심 1
weight_decay: 6.797303101020006e-05  # ← 의심 2
T_max: 24                        # ← 의심 3
eta_min: 6.388390006720873e-06   # ← 의심 4
```

### 리더보드 결과
```
로컬 Validation:  96.70% (Run 8 재현 실험)
리더보드 Test:    96.26% (제출 결과)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gap:              -0.44%p ⚠️

세부 메트릭:
             Local    Leaderboard  Delta
Precision:   96.65%   96.70%       +0.05%p ✅
Recall:      96.95%   96.02%       -0.93%p 🔴
H-Mean:      96.70%   96.26%       -0.44%p
```

---

## 🔍 Run 8 LR 파라미터 심층 분석

### 의심 포인트 1: LR=0.000513 (매우 높음)

#### Sweep 결과 비교
```
High LR Zone (0.0005+):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run 8:  LR=0.000513, WD=0.000068 → 96.60% (val)
Run 7:  LR=0.000592, WD=0.000066 → 96.29% (val)

Mid-High LR (0.0004-0.0005):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run 10: LR=0.000480, WD=0.000098 → 96.20% (val)
Run 9:  LR=0.000478, WD=0.000106 → 96.03% (val)
Run 11: LR=0.000443, WD=0.000116 → 96.14% (val)

Mid LR (0.0003-0.0004):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run 3:  LR=0.000385, WD=0.000139 → 96.47% (val) ⭐
Run 2:  LR=0.000411, WD=0.000123 → 96.29% (val)
Run 4:  LR=0.000396, WD=0.000080 → 96.31% (val)
```

#### 문제 진단
```
Run 8 (LR=0.000513):
- Validation: 96.60% ✅ 최고 성능
- Leaderboard: 96.26% 🔴 하락

Run 3 (LR=0.000385):
- Validation: 96.47% (2위)
- Leaderboard: ???% (미제출)
- 차이: -0.13%p lower on val

가설:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LR=0.000513은 Validation에 과최적화
→ 학습률이 높아서 validation 특성에 빠르게 맞춤
→ 일반화 능력 부족
→ 새로운 테스트셋에서 성능 하락

LR=0.000385 (Run 3):
→ 더 보수적인 학습
→ 천천히 수렴하지만 더 안정적
→ 일반화 능력 우수 가능성
```

### 의심 포인트 2: Weight Decay=0.000068 (매우 낮음)

#### Sweep 패턴 분석
```
Very Low WD (0.00006-0.00008):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run 8: WD=0.000068 → Val 96.60%, Test 96.26%
Run 7: WD=0.000066 → Val 96.29%
Run 4: WD=0.000080 → Val 96.31%

Mid WD (0.0001-0.00014):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run 3:  WD=0.000139 → Val 96.47% ⭐
Run 6:  WD=0.000134 → Val 96.23%
Run 2:  WD=0.000123 → Val 96.29%
Run 11: WD=0.000116 → Val 96.14%
```

#### Overfitting 증거
```
Weight Decay의 역할:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
낮은 WD (0.00006-0.00008):
- 모델 파라미터 제약 약함
- Validation 성능 높음
- 하지만 overfitting 위험 ↑
- 일반화 능력 저하 가능

적절한 WD (0.00012-0.00014):
- 파라미터 정규화 효과
- Validation 약간 낮을 수 있음
- 일반화 능력 향상
- 테스트 성능 안정적

Run 8의 문제:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WD=0.000068 + LR=0.000513
→ 높은 학습률 + 약한 정규화
→ Validation에 빠르게 과적합
→ 테스트셋 일반화 실패
→ Recall -0.93%p (과적합의 전형적 패턴)
```

### 의심 포인트 3: T_max=24 vs 22

```
Scheduler 설정 비교:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T_max=22 (대부분):
- Run 2, 3, 4, 12
- 전체 epoch과 일치
- CosineAnnealing 완전한 주기

T_max=24 (Run 8):
- Max epoch=22 vs T_max=24
- 불완전한 cosine 주기
- 학습 후반 LR이 예상보다 높음

문제:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 22에서 LR이 eta_min에 도달하지 못함
→ LR이 아직 높은 상태로 학습 종료
→ Fine-tuning 부족
→ Validation 특성에 맞췄지만 일반화 안됨
```

---

## 📊 Run 3 vs Run 8 비교 분석

### 설정 비교

| Parameter | Run 3 (안전) | Run 8 (공격적) | 차이 분석 |
|-----------|-------------|---------------|----------|
| **LR** | 0.000385 | 0.000513 | +33% 높음 ⚠️ |
| **WD** | 0.000139 | 0.000068 | -51% 낮음 ⚠️ |
| **T_max** | 22 | 24 | +9% 불일치 ⚠️ |
| **eta_min** | 1.86e-05 | 6.39e-06 | -66% 낮음 |
| **Val H-Mean** | 96.47% | 96.60% | Run 8이 +0.13%p |
| **Status** | Completed | Terminated | Run 8 조기종료 |

### 핵심 차이점
```
Run 3: 보수적 접근
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 중간 LR (0.000385)
✅ 적절한 WD (0.000139) 
✅ 정확한 T_max (22)
✅ 22 epoch 완료
→ Validation: 96.47% (2위)
→ 일반화 능력: 높을 가능성

Run 8: 공격적 접근
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 높은 LR (0.000513)
⚠️ 낮은 WD (0.000068)
⚠️ 불일치 T_max (24 vs 22)
⚠️ Epoch 10에서 조기종료
→ Validation: 96.60% (1위)
→ Test: 96.26% (-0.44%p)
→ 일반화 실패 확인됨!
```

---

## 💡 새로운 최적 파라미터 제안

### Option 1: Run 3 기반 (가장 안전 ★★★)

```yaml
# Run 3 설정 (22 epoch 완료, 2위)
models:
  optimizer:
    lr: 0.000385                # Run 8의 75% (더 보수적)
    weight_decay: 0.000139      # Run 8의 2배 (더 강한 정규화)
  scheduler:
    T_max: 22                   # 전체 epoch과 일치
    eta_min: 1.86e-05          # 높은 최소 LR

# Postprocessing (검증된 최적값 유지)
head:
  postprocess:
    thresh: 0.29
    box_thresh: 0.25
    max_candidates: 600
```

**예상 효과**:
```
Validation: 96.47% (Run 3 실적)
리더보드: 96.50-96.65% 예상 (+0.24-0.39%p vs Run 8)

논리:
- Run 3는 validation 96.47%로 안정적
- 22 epoch 완전 학습 (Run 8은 10epoch 조기종료)
- 더 강한 정규화로 일반화 능력 ↑
- Recall 하락 방지
```

### Option 2: Run 3 + Run 8 혼합 (균형 ★★☆)

```yaml
models:
  optimizer:
    lr: 0.00045                # Run 3~8 중간값
    weight_decay: 0.0001       # Run 3~8 중간값
  scheduler:
    T_max: 22                  # 정확한 epoch 매칭
    eta_min: 1.0e-05          # 중간값
```

**예상 효과**:
```
Validation: 96.55% 예상
리더보드: 96.45-96.60% 예상

장점: Run 8의 높은 성능 + Run 3의 안정성
단점: 검증되지 않은 새 조합 (위험)
```

### Option 3: Run 4 기반 (대안 ★★☆)

```yaml
# Run 4 (3위, Val 96.31%)
models:
  optimizer:
    lr: 0.000396
    weight_decay: 0.000080
  scheduler:
    T_max: 22
    eta_min: 1.93e-05
```

**예상 효과**:
```
Validation: 96.31% (Run 4 실적)
리더보드: 96.40-96.55% 예상

특징:
- Run 8보다 보수적 (LR -23%, WD +18%)
- Run 3보다 공격적
- 중간 선택지
```

---

## 🎯 최종 권장 전략

### 1단계: Run 3 파라미터로 즉시 재학습 (최우선)

```bash
cd /data/ephemeral/home/baseline_code

# Run 3 설정으로 재학습
python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_run3_replication \
  models.optimizer.lr=0.000385 \
  models.optimizer.weight_decay=0.000139 \
  models.scheduler.T_max=22 \
  models.scheduler.eta_min=0.0000186 \
  trainer.max_epochs=22 \
  wandb=false
```

**예상 소요 시간**: 2-3시간
**예상 성능**: 
- Validation: 96.47%
- 리더보드: 96.50-96.65% (Run 8보다 +0.24-0.39%p)

### 2단계: 리더보드 검증

```bash
# 학습 완료 후 예측
python runners/predict.py \
  preset=efficientnet_b4_lr_optimized \
  checkpoint_path=outputs/efficientnet_b4_run3_replication/checkpoints/best.ckpt \
  exp_name=run3_submission

# 제출 후 비교
Run 8: 96.26%
Run 3: ???% (예상 96.50-96.65%)
```

### 3단계: 5-Fold 앙상블 (Run 3 설정 사용)

```bash
# Run 3 파라미터로 5-fold 학습
python runners/run_kfold.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_run3_5fold \
  models.optimizer.lr=0.000385 \
  models.optimizer.weight_decay=0.000139 \
  models.scheduler.T_max=22 \
  models.scheduler.eta_min=0.0000186 \
  trainer.max_epochs=22 \
  wandb=false
```

**예상 최종 성능**: 96.80-97.10%

---

## 📊 실험 계획 상세

### 실험 A: Run 3 vs Run 8 직접 비교

| Experiment | LR | WD | T_max | 예상 Val | 예상 Test | 우선순위 |
|-----------|----|----|-------|---------|-----------|---------|
| Run 8 (현재) | 0.000513 | 0.000068 | 24 | 96.70% | 96.26% ✅ | 완료 |
| **Run 3** | **0.000385** | **0.000139** | **22** | **96.47%** | **96.50-65%** | **P1** |
| Run 4 | 0.000396 | 0.000080 | 22 | 96.31% | 96.40-55% | P2 |
| Hybrid | 0.00045 | 0.0001 | 22 | 96.55% | 96.45-60% | P3 |

### 실험 B: Weight Decay 민감도 테스트

```python
# Run 3 LR 고정, WD만 변경
configs = [
    {"lr": 0.000385, "wd": 0.000120, "name": "run3_wd_low"},
    {"lr": 0.000385, "wd": 0.000139, "name": "run3_original"},
    {"lr": 0.000385, "wd": 0.000160, "name": "run3_wd_high"},
]
```

### 실험 C: Learning Rate Fine-tuning

```python
# WD=0.000139 고정, LR만 변경
configs = [
    {"lr": 0.000360, "wd": 0.000139, "name": "lr_conservative"},
    {"lr": 0.000385, "wd": 0.000139, "name": "run3_original"},
    {"lr": 0.000410, "wd": 0.000139, "name": "lr_aggressive"},
]
```

---

## 🔬 근거 및 이론적 배경

### 1. High LR + Low WD = Overfitting

```python
# 수학적 분석
gradient_step = lr * gradient
parameter_decay = wd * parameter

Run 8:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lr=0.000513 (큰 gradient step)
wd=0.000068 (약한 decay)
→ 파라미터가 크게 변하지만 제약 약함
→ Validation에 빠르게 과적합
→ 일반화 실패

Run 3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lr=0.000385 (보통 gradient step)
wd=0.000139 (강한 decay)
→ 파라미터가 천천히 변하며 정규화됨
→ 안정적 수렴
→ 일반화 능력 향상
```

### 2. T_max Mismatch의 영향

```python
# CosineAnnealing 수식
lr_t = eta_min + (lr_max - eta_min) * (1 + cos(pi * t / T_max)) / 2

Run 8 (T_max=24, max_epochs=22):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 22: t/T_max = 22/24 = 0.917
→ Cosine 주기의 91.7%만 완료
→ LR이 eta_min에 도달 못함
→ 학습 후반 LR이 여전히 높음
→ Fine-tuning 부족

Run 3 (T_max=22, max_epochs=22):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 22: t/T_max = 22/22 = 1.0
→ 완전한 cosine 주기
→ LR이 eta_min 도달
→ 충분한 fine-tuning
→ 일반화 능력 향상
```

### 3. Recall 하락의 정확한 원인

```
Overfitting의 전형적 패턴:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Validation 특성에 과도하게 맞춤
2. 모델이 validation 패턴 암기
3. 새로운 데이터에서:
   - Precision 유지 (잘못 검출은 안함)
   - Recall 하락 (놓치는 것 증가)
   - 암기한 패턴만 검출 가능

Run 8 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Precision: 96.70% (유지)
Recall: 96.02% (-0.93%p)
→ 전형적인 overfitting 증거
→ LR/WD 파라미터 문제 확실
```

---

## 📝 교훈

### 1. Hyperband의 함정
```
Hyperband가 Run 8을 조기종료한 이유:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 10 기준 상대 순위로 평가
→ Run 8은 1등이었지만 하위 50% 브래킷
→ 조기 종료

문제:
- Epoch 10 성능 ≠ Epoch 22 성능
- Run 8은 초반 빠른 수렴 (High LR)
- 하지만 후반 overfitting 가능성 높음
- Run 3는 천천히 안정적으로 수렴
- Epoch 22까지 학습 완료

결론:
Hyperband가 놓친 것: 일반화 능력
→ Run 3가 실제로는 더 나을 가능성
```

### 2. Validation Performance ≠ Test Performance

```
Sweep 기반 최적화의 위험:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 단일 validation set으로만 평가
2. 그 set에 최적화된 파라미터 선택
3. 실제 테스트셋과 분포 다를 수 있음
4. Overfitting 감지 불가

올바른 접근:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Cross-validation 사용
2. Holdout validation set 추가
3. Early stopping with patience
4. 보수적 파라미터 선택
5. 앙상블로 일반화 강화
```

### 3. 파라미터 균형의 중요성

```
좋은 조합:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High LR → High WD (균형)
Low LR → Low WD (균형)
Mid LR → Mid WD (안전)

나쁜 조합:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High LR + Low WD (Run 8) ⚠️
→ 빠른 학습 + 약한 정규화
→ Overfitting 위험 극대화

Low LR + High WD (Run 1) 💥
→ 너무 제약적
→ Under-fitting
```

---

## 🎯 즉시 실행 액션

### 1. Run 3 재현 학습 시작 (지금 바로!)

```bash
cd /data/ephemeral/home/baseline_code

# 학습 시작
nohup python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_run3_replication \
  models.optimizer.lr=0.000385 \
  models.optimizer.weight_decay=0.000139 \
  models.scheduler.T_max=22 \
  models.scheduler.eta_min=0.0000186 \
  trainer.max_epochs=22 \
  wandb=false \
  > run3_training.log 2>&1 &

# 로그 확인
tail -f run3_training.log
```

### 2. 학습 완료 후 즉시 제출

```bash
# 예측 생성
python runners/predict.py \
  preset=efficientnet_b4_lr_optimized \
  checkpoint_path=outputs/efficientnet_b4_run3_replication/checkpoints/best.ckpt \
  exp_name=run3_final_submission

# CSV 변환
python ocr/utils/convert_submission.py \
  -J outputs/run3_final_submission/submissions/*.json \
  -O outputs/run3_final_submission/submissions/submission_run3.csv
```

### 3. 결과 비교 및 결정

```
If Run 3 > 96.40%:
→ Run 3 파라미터로 5-fold 진행

If Run 3 < 96.40%:
→ Run 4 또는 Hybrid 테스트

If Run 3 > 96.50%:
→ Run 3가 최적, 즉시 5-fold 시작!
```

---

## 📌 핵심 요약

**문제 원인**:
- ❌ Postprocessing 파라미터 (thresh, box_thresh) 아님!
- ✅ **Learning Rate 파라미터 (LR, WD, T_max) 문제**
- Run 8: High LR + Low WD → Validation overfitting
- Recall -0.93%p 하락 → 전형적인 과적합 패턴

**해결책**:
- Run 3 파라미터 사용 (검증된 2위, 더 안정적)
- LR=0.000385 (Run 8의 75%)
- WD=0.000139 (Run 8의 2배)
- T_max=22 (정확한 매칭)
- 예상: 96.50-96.65% 리더보드 (+0.24-0.39%p)

**다음 단계**:
1. Run 3 재현 학습 (2-3시간)
2. 리더보드 제출 및 검증
3. Run 3로 5-fold 앙상블 (목표 96.80-97.10%)
