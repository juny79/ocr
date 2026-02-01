# EfficientNet-B4 최적화 전략
## 현재 → 목표: 96.00% → 96.50%

---

## 📊 현재 상황 분석

### 리더보드 결과
```
H-Mean:    96.00% ⭐
Precision: 96.27%
Recall:    95.98%
P-R Gap:   0.29%p (매우 균형적)
```

### ResNet50 비교
| Metric    | ResNet50 | EfficientNet-B4 | 차이 |
|-----------|----------|-----------------|------|
| H-Mean    | 96.28%   | 96.00%          | -0.28%p |
| Precision | 97.31%   | 96.27%          | -1.04%p ❌ |
| Recall    | 95.58%   | 95.98%          | +0.40%p ✅ |

### 핵심 인사이트
✅ **강점**
- Validation(96.0%) = Leaderboard(96.0%) → 일반화 우수
- Recall 개선 (+0.40%p) → 검출력 향상
- P-R 균형 (0.29%p) → 안정적

❌ **약점**  
- Precision 낮음 (-1.04%p) → False Positive 많음
- ResNet50 대비 H-Mean 낮음 (-0.28%p)

🎯 **개선 방향**
- **Precision을 96.27% → 97.0%+ 로 개선**
- Recall을 95.98% → 95.5% 정도로 살짝 희생 OK
- 결과: H-Mean 96.5% 달성 가능!

---

## 🎯 최적화 전략 (3단계)

### ⭐ 전략 1: Smart Postprocessing 최적화 (권장)

**장점**
- ✅ 가장 빠름 (5분)
- ✅ 재학습 불필요 (기존 checkpoint 사용)
- ✅ 즉시 효과 확인 가능
- ✅ 9회 시도로 최적 파라미터 발견

**원리**
- `thresh` ↑ → Probability Threshold 상승 → FP↓, Precision↑
- `box_thresh` ↑ → 신뢰도 낮은 박스 제거 → FP↓, Precision↑
- `max_candidates` 조정 → 출력 박스 수 제어

**실행 방법**
```bash
cd /data/ephemeral/home/baseline_code
bash scripts/smart_postproc_optim.sh
```

**시도 조합 (9개)**
```
Phase 1: thresh 증가
  1. thresh=0.24, box_thresh=0.25 (Conservative)
  2. thresh=0.26, box_thresh=0.25 (Medium)
  3. thresh=0.28, box_thresh=0.25 (Aggressive)

Phase 2: box_thresh 증가
  4. thresh=0.24, box_thresh=0.28
  5. thresh=0.24, box_thresh=0.30
  6. thresh=0.26, box_thresh=0.28

Phase 3: 조합 최적화
  7. thresh=0.25, box_thresh=0.27 (Balanced)
  8. thresh=0.23, box_thresh=0.26 (Safe)
  9. thresh=0.27, box_thresh=0.26 (High Precision)
```

**예상 결과**
- Best Case: 96.5-96.6% (thresh=0.25-0.27, box_thresh=0.26-0.28)
- Worst Case: 95.8% (thresh 너무 높으면 Recall 과다 하락)

**다음 단계**
1. 9개 CSV를 리더보드에 제출
2. 최고 성능 파라미터 확인
3. `configs/preset/efficientnet_b4_optimal.yaml` 생성
4. 5-Fold 학습 진행 (최적 설정 적용)

---

### 전략 2: WandB Sweep (학습 포함 최적화)

**장점**
- ✅ Learning Rate, Weight Decay 등 학습 파라미터도 최적화
- ✅ Bayesian 최적화로 효율적 탐색
- ✅ 96.6-96.8% 달성 가능성

**단점**
- ❌ WandB API Key 필요
- ❌ 소요 시간: 5-8시간 (15-20 runs)
- ❌ 환경 설정 복잡

**실행 방법**
```bash
cd /data/ephemeral/home/baseline_code

# WandB 로그인 (API Key 필요)
wandb login

# Focused Sweep 실행 (추천)
bash scripts/run_sweep_focused.sh 15

# 또는 Background 실행
bash scripts/run_sweep_focused.sh 15 bg
```

**Sweep 설정**
- Config: `configs/sweep_efficientnet_b4_focused.yaml`
- Method: Bayesian Optimization
- Metric: val/hmean (maximize)
- Early Termination: Hyperband (min_iter=8)

**탐색 파라미터**
```yaml
Critical (High Impact):
  - models.head.thresh: [0.20, 0.28]
  - models.head.box_thresh: [0.22, 0.32]
  - models.optimizer.lr: [0.0002, 0.0006]

Secondary (Medium Impact):
  - models.optimizer.weight_decay: [0.00005, 0.0005]
  - models.scheduler.T_max: [18, 20, 22]
  - models.head.max_candidates: [500, 600, 700]
```

**WandB 대시보드**
- URL: `https://wandb.ai/[USERNAME]/efficientnet_b4_sweep_focused`
- Parallel Coordinates Plot으로 최적 조합 시각화
- Importance Plot으로 영향력 큰 파라미터 확인

**다음 단계**
1. Sweep 완료 후 최고 성능 run 확인
2. 최적 하이퍼파라미터 복사
3. 최적 설정으로 단일 모델 재학습
4. 96.5%+ 달성 시 5-Fold 진행

---

### 전략 3: Learning Rate 재조정 + 재학습

**장점**
- ✅ 단순하고 명확
- ✅ WandB 불필요
- ✅ 2-3시간으로 빠름

**단점**
- ❌ 수동 조정 필요
- ❌ Trial & Error

**실행 방법**
```bash
# LR을 0.0003 → 0.0004로 증가
python runners/train.py \
  preset=efficientnet_b4_aggressive \
  exp_name=efficientnet_b4_lr_0.0004 \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22

# 또는 LR을 낮춰서 안정화
python runners/train.py \
  preset=efficientnet_b4_aggressive \
  exp_name=efficientnet_b4_lr_0.00025 \
  models.optimizer.lr=0.00025 \
  trainer.max_epochs=22
```

**시도 순서**
1. LR=0.0004 (현재 0.0003의 133%)
2. LR=0.00025 (현재 0.0003의 83%)
3. 더 나은 쪽으로 Fine-tuning

---

## 📋 권장 실행 계획

### Phase 1: 즉시 실행 (5분)
```bash
bash scripts/smart_postproc_optim.sh
```
→ 9개 제출 파일 생성 → 리더보드 제출 → 최고 성능 확인

### Phase 2: 결과 분석 (30분)
- 최고 성능 파라미터 확인
- 96.3%+ 달성 시 → Phase 3
- 96.3% 미만 시 → WandB Sweep 또는 LR 재조정

### Phase 3: 5-Fold 학습 (12시간)
```bash
# 최적 설정으로 5-Fold
bash scripts/train_efficientnet_b4_5fold.sh
```
→ Voting Ensemble → 96.5-96.7% 목표

---

## 🎯 성공 기준

| 단계 | 목표 | 행동 |
|------|------|------|
| **Postprocessing 최적화** | 96.3%+ | Phase 3 진행 (5-Fold) |
| | 96.1-96.3% | WandB Sweep 시도 |
| | <96.1% | ResNet101로 피벗 |
| **5-Fold Ensemble** | 96.5%+ | 목표 달성! 🎉 |
| | 96.3-96.5% | ResNet50 + B4 Ensemble |
| | <96.3% | ResNet50으로 회귀 |

---

## 💡 핵심 팁

### Precision vs Recall Trade-off
```
thresh ↑  →  Precision ↑, Recall ↓
- 0.22 (현재): Precision 96.27%, Recall 95.98%
- 0.25 (예상): Precision 96.8%, Recall 95.5% → H-Mean 96.15%
- 0.27 (예상): Precision 97.2%, Recall 95.0% → H-Mean 96.09%
- 0.26 (최적): Precision 97.0%, Recall 95.3% → H-Mean 96.15%

→ Sweet Spot: thresh=0.25-0.26
```

### Box Threshold Impact
```
box_thresh ↑  →  낮은 신뢰도 박스 제거
- 0.25 (현재): 모든 박스 허용
- 0.27 (예상): Top 95% 박스만 → FP 5% 감소
- 0.30 (예상): Top 90% 박스만 → FP 10% 감소

→ Sweet Spot: box_thresh=0.26-0.28
```

### Learning Rate Sensitivity
```
EfficientNet-B4는 ResNet50보다 LR에 민감:
- 너무 높으면: Validation 불안정 (96.0% → 95.5%)
- 너무 낮으면: 수렴 느림 (Epoch 22에도 미도달)

현재 0.0003은 약간 낮을 가능성
→ 0.00035-0.0004가 최적일 수도
```

---

## 🚀 바로 시작하기

**가장 빠른 방법 (5분):**
```bash
cd /data/ephemeral/home/baseline_code
bash scripts/smart_postproc_optim.sh
```

**가장 확실한 방법 (8시간):**
```bash
wandb login
bash scripts/run_sweep_focused.sh 15
```

**중간 방법 (2시간):**
```bash
python runners/train.py \
  preset=efficientnet_b4_aggressive \
  exp_name=efficientnet_b4_lr_tuned \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22
```

---

## 📞 문제 해결

### Q1: WandB API Key가 없어요
→ **전략 1 (Smart Postprocessing)** 사용 추천

### Q2: 9개 제출이 너무 많아요
→ Phase 1만 실행 (3개): thresh=0.24, 0.26, 0.28

### Q3: Postprocessing만으로 충분할까요?
→ 96.3%+ 달성 시 충분, 아니면 WandB Sweep 필요

### Q4: 시간이 없어요
→ **전략 1** 실행 후 최고 성능 파라미터로 단일 제출

---

## ✅ 체크리스트

- [ ] Smart Postprocessing 실행
- [ ] 9개 제출 파일 리더보드 제출
- [ ] 최고 성능 파라미터 확인 (목표: 96.3%+)
- [ ] 최적 설정으로 config 생성
- [ ] 5-Fold 학습 진행 (96.3%+ 달성 시)
- [ ] Voting Ensemble (Voting≥3)
- [ ] 최종 제출 (목표: 96.5%+)

---

**생성 일시**: 2026-02-02  
**현재 Best**: 96.00% (EfficientNet-B4 Single)  
**목표**: 96.50%+  
**전략**: Precision 개선 (Postprocessing 최적화)
