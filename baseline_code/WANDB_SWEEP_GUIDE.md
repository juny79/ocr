# WandB Sweep을 이용한 HRNet-W44 1280x1280 최적 파라미터 탐색

## 📊 현재 성과
- **제출 결과**: H-Mean 97.14%, Precision 97.35%, Recall 97.08%
- **현재 파라미터**:
  - lr: 0.00045
  - weight_decay: 0.00006
  - T_max: 20
  - eta_min: 0.000008

---

## 🎯 Sweep 목표

더 나은 파라미터 조합을 자동으로 찾기 위해 **Bayesian Optimization**과 **Hyperband** 조기 종료를 사용합니다.

### 탐색 범위

| 파라미터 | 탐색 범위 | 현재값 | 비고 |
|---------|---------|--------|------|
| **Learning Rate** | 0.00001 ~ 0.0002 | 0.00045 | log scale (주변 ±2배) |
| **Weight Decay** | 0.0000061 ~ 0.000123 | 0.00006 | log scale (±2배) |
| **T_max** | [15, 18, 20, 25] | 20 | 정수값 선택 |
| **eta_min** | 0.0000022 ~ 0.000045 | 0.000008 | log scale (±5배) |

### 최적화 전략
- **방식**: Bayesian Optimization (스마트 탐색)
- **메트릭**: val/hmean 최대화
- **조기 종료**: Hyperband (5 epoch 후 성능 낮은 조합 자동 중단)
- **예상 시간**: 8 parallel runs × ~6시간/run = 병렬 실행 시 ~6시간

---

## 🚀 실행 방법

### 방법 1: 자동 실행 (권장)
```bash
cd /data/ephemeral/home/baseline_code
chmod +x run_sweep.sh
./run_sweep.sh
```

이 명령어는:
1. ✅ Sweep 설정 초기화
2. ✅ Sweep ID 생성
3. ✅ 8개 병렬 에이전트 시작

### 방법 2: 수동 실행 (단계별)

**Step 1: Sweep 초기화**
```bash
cd /data/ephemeral/home/baseline_code
wandb sweep sweep_hrnet_w44_1280.yaml \
  --project hrnet-w44-1280-sweep \
  --entity juny79
```

출력 예시:
```
Create sweep with ID: abc123xyz
Run sweep agent with: wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz
```

**Step 2: Sweep 에이전트 실행** (터미널 1에서)
```bash
cd /data/ephemeral/home/baseline_code
wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz --count 8
```

또는 **병렬 실행** (여러 터미널에서 동시 실행):
```bash
# 터미널 1, 2, 3... 에서 각각 실행
wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz
```

---

## 📈 실시간 모니터링

### WandB Dashboard 확인
```
https://wandb.ai/juny79/hrnet-w44-1280-sweep
```

Dashboard에서 확인 가능한 정보:
- 각 실행의 학습 곡선 (loss, val_hmean 등)
- 파라미터 vs 성능 관계
- 최고 성능 조합
- 병렬 실행 진행도

### 로컬에서 실시간 확인
```bash
# Sweep 상태 확인
wandb sweep status juny79/hrnet-w44-1280-sweep/abc123xyz

# 최신 결과 확인
wandb sweeps best juny79/hrnet-w44-1280-sweep
```

---

## 💡 Bayesian Optimization 이해하기

```
초기 실행 (3-4개):
  → 파라미터 공간 탐험

중기 실행 (5-6개):
  → 좋은 영역으로 집중
  → 조기 종료 활용

후기 실행 (7-8개):
  → 최고 성능 조합 근처 탐색
  → 수렴 확인
```

---

## ⚡ 조기 종료 (Hyperband) 메커니즘

```
각 실행의 5 epoch마다 검사:

Epoch 5:  낮은 성능 → 중단 (24시간 절약)
Epoch 10: 중간 성능 → 계속 진행
Epoch 15: 높은 성능 → 계속 진행
Epoch 20: 최종 성능 기록
```

**효과**: 나쁜 파라미터 조합은 조기에 중단되어 리소스 절약

---

## 🎓 예상 결과

Sweep 완료 후 WandB에서 자동 생성되는 보고서:

```
최고 성능 설정:
  lr: 0.0003 (또는 다른 값)
  weight_decay: 0.00005
  T_max: 20
  eta_min: 0.000012
  
예상 H-Mean: 97.20% ~ 97.35%
```

---

## 📊 Sweep 설정 상세 (sweep_hrnet_w44_1280.yaml)

### 파라미터 설정 이유

**Learning Rate - log_uniform**
- 현재 0.00045가 좋은 값이므로 주변에서 탐색
- Log scale: 0.00001 ~ 0.0002 (현재값의 ±2배 범위)
- Bayesian Optimization이 자동으로 유망한 영역 탐색

**Weight Decay - log_uniform**
- 배치 크기 2에 맞추어 조정
- 현재 0.00006 주변에서 ±2배 범위 탐색

**T_max - discrete**
- 코사인 어닐링 사이클
- 15, 18, 20, 25 중 최적값 선택
- 범주형 검색 (더 빠른 수렴)

**eta_min - log_uniform**
- 최소 학습율
- ±5배 범위로 더 넓게 탐색

---

## 🔧 트러블슈팅

### 문제 1: 메모리 부족
```bash
# Parallel runs 감소
wandb agent ... --count 4  # 8에서 4로 감소
```

### 문제 2: Sweep 중단되었을 때
```bash
# 동일한 sweep ID로 다시 시작
wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz --count 4
```

### 문제 3: 최적값 찾기
```bash
# WandB에서 자동 제시 (Dashboard의 "Best" 표시)
# 또는 프로그래밍으로:
wandb sweeps best juny79/hrnet-w44-1280-sweep
```

---

## 📌 Sweep 완료 후 다음 단계

1. **최고 파라미터 확인**
   - WandB Dashboard에서 최고 H-Mean 찾기
   - 모든 fold에 적용할 파라미터 결정

2. **다른 fold에 적용**
   ```bash
   # Fold 1-4도 동일한 파라미터로 학습
   python runners/train.py preset=hrnet_w44_1280 \
     models.optimizer.lr=<best_lr> \
     models.optimizer.weight_decay=<best_wd> \
     models.scheduler.T_max=<best_tmax> \
     models.scheduler.eta_min=<best_etamin> \
     trainer.max_epochs=20
   ```

3. **5-fold 앙상블**
   ```bash
   python scripts/ensemble_kfold.py
   ```

---

## 📝 예상 일정

```
시작: 지금
Step 1 (Sweep 초기화): 5분
Step 2 (병렬 실행): ~6시간 (8개 run 동시)
Step 3 (최고값 분석): 10분
Step 4 (Fold 1-4 학습): ~30시간 (병렬 실행 시 ~10시간)
Step 5 (5-fold 앙상블): 1시간

총 예상 시간: 48시간 이내 (모든 fold 포함)
```

---

## 🎯 성공 지표

✅ Sweep 완료 시 다음 확인:
- [ ] 최고 H-Mean > 97.14% (현재값)
- [ ] 모든 run이 안정적으로 완료됨
- [ ] 파라미터 영향도 시각화 (WandB 제공)
- [ ] 최적값 조합 도출

**예상 최종 성과**: H-Mean **97.20% ~ 97.40%** 🚀
