# WandB Sweep 빠른 시작 가이드

## 📊 현재 성과
- **리더보드 점수**: H-Mean **97.14%** (Precision 97.35%, Recall 97.08%)
- **향상도**: 기준 96.44%에서 **+0.70% ⬆️**

---

## 🎯 목표
WandB Sweep을 사용하여 최적의 파라미터 조합을 자동으로 찾기

---

## 🚀 3단계 실행 방법

### 1️⃣ 터미널 열기
```bash
cd /data/ephemeral/home/baseline_code
```

### 2️⃣ Sweep 초기화 및 실행
```bash
chmod +x start_sweep.sh
./start_sweep.sh
```

**자동으로:**
- ✅ Sweep 설정 초기화
- ✅ Sweep ID 생성
- ✅ 에이전트 실행 옵션 선택

### 3️⃣ 실시간 모니터링
```
https://wandb.ai/juny79/hrnet-w44-1280-sweep
```

---

## 🔧 수동 실행 (step-by-step)

### Sweep 초기화만
```bash
cd /data/ephemeral/home/baseline_code
wandb sweep sweep_hrnet_w44_1280.yaml \
  --project hrnet-w44-1280-sweep \
  --entity juny79
```

 예시:
```
Create sweep with ID: abc123xyz
Run sweep agent with: wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz
```

### 에이전트 실행 (병렬 - 권장)
```bash
wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz --count 8
```

**또는 여러 터미널에서 동시 실행:**
```bash
# 터미널 1, 2, 3...
wandb agent juny79/hrnet-w44-1280-sweep/abc123xyz
```

---

## ⏱️ 예상 시간

| 단계 | 소요 시간 |
|------|---------|
| Sweep 초기화 | 5분 |
| 병렬 실행 (8개) | ~6시간 |
| 최적값 분석 | 10분 |
| **총합** | **~6.5시간** |

---

## 📈 탐색 파라미터

| 파라미터 | 탐색 범위 | 현재값 |
|---------|---------|--------|
| Learning Rate | 0.00001 ~ 0.0002 | 0.00045 |
| Weight Decay | 0.0000061 ~ 0.000123 | 0.00006 |
| T_max | [15, 18, 20, 25] | 20 |
| eta_min | 0.0000022 ~ 0.000045 | 0.000008 |

---

## 💡 최적화 방식

- **방법**: Bayesian Optimization (스마트 탐색)
- **목표 메트릭**: val/hmean 최대화
- **조기 종료**: Hyperband (5 epoch 후 낮은 성능 자동 중단)

---

## ✅ Sweep 완료 후

1. **최고 성능 파라미터 확인**
   - WandB Dashboard에서 "Best" 표시된 run 확인

2. **모든 Fold에 적용**
   ```bash
   python runners/train.py preset=hrnet_w44_1280 \
     models.optimizer.lr=<최고값_lr> \
     models.optimizer.weight_decay=<최고값_wd> \
     models.scheduler.T_max=<최고값_tmax> \
     models.scheduler.eta_min=<최고값_etamin>
   ```

3. **5-Fold 앙상블**
   ```bash
   python scripts/ensemble_kfold.py
   ```

---

## 📊 예상 결과

- **최고 H-Mean**: 97.20% ~ 97.40% (현재 97.14%에서 향상)
- **성능 향상도**: +0.06% ~ +0.26%

---

## 🆘 트러블슈팅

### 메모리 부족
```bash
# 병렬 실행 수 감소
wandb agent <SWEEP_ID> --count 4  # 8 → 4
```

### Sweep 중단 후 재개
```bash
# 동일한 Sweep ID로 다시 시작
wandb agent juny79/hrnet-w44-1280-sweep/<ID> --count 4
```

---

## 📚 추가 정보

 설정은 다음 파일 참고:
- [WANDB_SWEEP_GUIDE.md](WANDB_SWEEP_GUIDE.md)
- [sweep_hrnet_w44_1280.yaml](sweep_hrnet_w44_1280.yaml)

---

**이제 시작할 준비가 되었습니다! 🚀**

```bash
cd /data/ephemeral/home/baseline_code
./start_sweep.sh
```
