# W&B (Weights & Biases) 실험 추적 설정 가이드

## 📊 개요

W&B는 머신러닝 실험 추적, 시각화, 비교를 위한 플랫폼입니다. 
이 프로젝트에서 학습 과정을 실시간으로 모니터링할 수 있습니다.

---

## 🚀 설정 단계

### Step 1: W&B 회원가입

1. **W&B 웹사이트 접속**: https://wandb.ai
2. **회원가입** (GitHub 계정으로 회원가입 권장)
3. **API Key 확인**: https://wandb.ai/settings/keys

### Step 2: 로컬 환경에 W&B 로그인

```bash
# W&B 설치 (이미 requirements.txt에 포함됨)
pip install wandb

# W&B 로그인
wandb login

# API Key 입력 (위에서 확인한 키)
# 또는 환경 변수로 설정
export WANDB_API_KEY="your-api-key"
```

### Step 3: 설정 파일 수정

**Option A: 명령어로 실행 (W&B 활성화)**

```bash
cd baseline_code
python runners/train.py preset=example wandb=True
```

**Option B: YAML 설정 수정**

`configs/train.yaml` 파일에서:

```yaml
wandb: True  # False → True
project_name: "ocr-receipt-detection"
```

---

## 📈 W&B 추적되는 메트릭

### 자동 추적

- **손실 함수**
  - `train/loss` - 학습 손실
  - `train/loss_prob` - Probability map 손실
  - `train/loss_thresh` - Threshold map 손실
  - `train/loss_binary` - Binary map 손실
  - `val/loss` - 검증 손실

- **평가 지표**
  - `val/recall` - Recall (재현율)
  - `val/precision` - Precision (정확도)
  - `val/hmean` - H-Mean (F1-Score)

- **학습 상태**
  - Learning Rate
  - Epoch 진행률
  - GPU/CPU 사용량
  - 학습 시간

### 로깅된 설정값

- Optimizer (Adam)
- Learning Rate (0.001)
- Batch Size (16)
- Epochs (10)
- 모든 하이퍼파라미터

---

## 🔧 고급 설정

### 커스텀 메트릭 로깅

`ocr/lightning_modules/ocr_pl.py`에서 추가 커스텀 로깅:

```python
# 현재 이미 구현됨
self.log('val/recall', recall, on_epoch=True, prog_bar=True)
self.log('val/precision', precision, on_epoch=True, prog_bar=True)
self.log('val/hmean', hmean, on_epoch=True, prog_bar=True)
```

### 체크포인트 저장

W&B 설정에서 `log_model=True`로 설정되어 있어, 모든 체크포인트가 자동 저장됩니다.

```yaml
# runners/train.py에서
logger = WandbLogger(
    project="ocr-receipt-detection",
    name=exp_name,
    log_model=True,  # ← 모델 자동 저장
    tags=["baseline", "dbnet"],
)
```

---

## 📊 W&B 대시보드 사용법

### 1. 실험 비교

```
W&B Dashboard:
├── Projects
│   └── ocr-receipt-detection
│       ├── Runs (각 학습 실행)
│       ├── Comparing Runs (여러 실행 비교)
│       └── Artifacts (모델, 데이터)
```

### 2. 실시간 모니터링

- **Graphs**: 손실, 메트릭 실시간 그래프
- **System**: GPU, CPU, 메모리 사용량
- **Logs**: 콘솔 출력 로그

### 3. 하이퍼파라미터 스윕 (Sweep)

여러 하이퍼파라미터를 자동으로 테스트:

```bash
# Sweep 설정 YAML 생성 후
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

---

## 🎯 예상 W&B 대시보드 구성

```
Run: exp_20260130_1430
├── Metrics
│   ├── train/loss
│   ├── val/loss
│   ├── val/recall ⬆️ (목표)
│   ├── val/precision
│   └── val/hmean
├── System
│   ├── GPU Utilization
│   ├── Memory Usage
│   └── Training Time
└── Artifacts
    ├── model-epoch-01.ckpt
    ├── model-epoch-02.ckpt
    └── model-epoch-03.ckpt
```

---

## 💡 Recall 개선 모니터링

W&B에서 다음을 추적하여 개선 상황을 확인:

1. **Baseline 실행 (기존 파라미터)**
   ```bash
   python runners/train.py preset=example wandb=True exp_name="baseline_v1"
   ```

2. **Tuning 실행 (개선된 파라미터)**
   ```bash
   python runners/train.py preset=example wandb=True exp_name="tuned_postprocess_v1"
   ```

3. **W&B Dashboard에서 비교**
   - 두 Run을 선택
   - "Compare" 클릭
   - Recall 개선도 시각화

---

## 🔐 보안 주의사항

- ⚠️ API Key를 공개 저장소에 커밋하지 마세요
- `.gitignore`에 `wandb/` 폴더 제외 (이미 설정됨)
- 프라이빗 프로젝트 사용 권장

---

## 🐛 트러블슈팅

### 문제: "wandb: ERROR not authenticated"

해결책:
```bash
wandb login
# 또는
export WANDB_API_KEY="your-api-key"
```

### 문제: "Project not found"

해결책:
```bash
# W&B 웹사이트에서 프로젝트 생성 후 실행
python runners/train.py wandb=True project_name="ocr-receipt-detection"
```

### 문제: 오프라인 모드 (인터넷 없을 때)

```bash
export WANDB_MODE=offline
python runners/train.py wandb=True

# 나중에 온라인 상태에서 동기화
wandb sync /path/to/run
```

---

## 📚 참고 자료

- **W&B 공식 문서**: https://docs.wandb.ai
- **PyTorch Lightning + W&B**: https://docs.wandb.ai/guides/integrations/lightning
- **W&B API Reference**: https://docs.wandb.ai/ref/python

---

## ✅ 현재 구현 상태

| 기능 | 상태 |
|------|------|
| **WandbLogger 통합** | ✅ 구현 |
| **자동 메트릭 로깅** | ✅ 구현 |
| **하이퍼파라미터 저장** | ✅ 구현 |
| **모델 아티팩트 저장** | ✅ 구현 |
| **학습 로그** | ✅ 구현 |
| **하이퍼파라미터 스윕** | 🔄 선택사항 |

---

**W&B를 활용하여 실험을 효율적으로 추적하고 관리하세요!** 🚀
