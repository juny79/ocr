# 🚀 WandB Sweep 최적 파라미터 탐색 실행 완료

## Sweep 설정 정보

- **Sweep ID**: `bw1bjr3b`
- **프로젝트**: `fc_bootcamp/ocr-receipt-detection`
- **방법**: Bayes Optimization
- **목표 메트릭**: `val_h_mean` (최대화)
- **최대 시도**: 10개

## 탐색 범위

### 학습 파라미터
- **Learning Rate**: 0.0008 ~ 0.002
- **Weight Decay**: 0.0001 ~ 0.0006
- **T_max (스케줄러)**: 8 ~ 15

### 후처리 파라미터
- **Detection Threshold**: 0.2 ~ 0.24
- **Box Threshold**: 0.4 ~ 0.44

### 모델 설정
- **배치 크기**: [8, 16, 32]
- **에포크**: [10, 13, 15]
- **모델 프리셋**: hrnet_w44_1024

## 진행 상황

Sweep agent가 현재 실행 중입니다. 각 시도는 약 15-20분 소요됩니다.

### 실시간 모니터링
- WandB 대시보드: https://wandb.ai/fc_bootcamp/ocr-receipt-detection/sweeps/bw1bjr3b
- 로컬 로그: `/data/ephemeral/home/baseline_code/sweep_final.log`

### 모니터링 명령어
```bash
# 진행 상황 실시간 확인
tail -f /data/ephemeral/home/baseline_code/sweep_final.log

# 프로세스 상태 확인
ps aux | grep "train.py"

# 결과 분석 (완료 후)
cd /data/ephemeral/home/baseline_code
source /data/ephemeral/home/venv/bin/activate
export WANDB_API_KEY=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
python analyze_sweep.py
```

## 다음 단계

1. **Sweep 완료 대기** (약 2-3시간)
2. **최적 파라미터 추출**
3. **최종 모델 학습** (최적 파라미터로)
4. **리더보드 제출**

---

**참고**: 
- 각 trial은 10-15 에포크 학습하므로 시간이 걸립니다
- WandB 대시보드에서 실시간으로 성능 추이를 확인할 수 있습니다
- Bayes Optimization은 이전 시도 결과를 기반으로 다음 파라미터를 선정합니다
