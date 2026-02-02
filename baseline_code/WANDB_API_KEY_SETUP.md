# WandB API Key 설정 가이드

## ❌ 문제: API Key가 올바르지 않습니다

현재 설정된 API Key는 더미 값입니다 (17자, 실제는 40자 필요)

## ✅ 해결 방법

### 1단계: WandB API Key 확인

1. WandB 웹사이트 방문: https://wandb.ai
2. 로그인
3. Settings → API Keys 메뉴
4. API Key 복사 (40자 길이)

### 2단계: API Key 설정

터미널에서 다음 명령 실행:

```bash
export WANDB_API_KEY='your-40-character-api-key-here'
```

**주의**: 따옴표 안에 실제 40자 API Key를 붙여넣으세요!

### 3단계: 확인

```bash
echo ${#WANDB_API_KEY}
```

출력이 **40**이어야 합니다.

### 4단계: Sweep 재실행

```bash
cd /data/ephemeral/home/baseline_code
python scripts/run_sweep_python.py 12
```

---

## 대안: WandB 없이 수동 LR 테스트

만약 WandB 설정이 어렵다면:

```bash
cd /data/ephemeral/home/baseline_code
python runners/train.py \
  preset=efficientnet_b4_lr_optimized \
  exp_name=efficientnet_b4_lr_0.0004 \
  models.optimizer.lr=0.0004 \
  trainer.max_epochs=22
```

- 소요 시간: 2시간
- 예상 성능: 96.45-96.55%
- Sweep보다 빠르고 간단함
