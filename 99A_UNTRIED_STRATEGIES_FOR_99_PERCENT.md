# 미시도 전략 종합 보고서: 98.63% → 99%+ 달성 로드맵

**작성일**: 2026-02-13 (v2.0 업데이트)  
**현재 최고 점수**: H-Mean **98.63%** (HRNet-W44 1024px + External Data + K-Fold Fold3)  
**최종 목표**: H-Mean **99.0%+** 달성  
**분석 기간**: 00~56번 보고서 분석 + 추가 전략 4개 통합  
**🆕 v2.0 신규 추가**: Tiny Box Loss, 13K Pre-training, 2단계 학습, P2 FPN  

---

## 📊 Executive Summary

### 현재 상황
- **달성한 개선**: 88.18% → 98.63% (+10.45%p)
- **실행된 6단계 모멘텀**:
  1. 후처리 조정 (+4.30%p)
  2. ResNet18 → ResNet50 (+3.72%p)
  3. Grid Search (+0.33%p)
  4. HRNet-W44 1280px (+1.27%p)
  5. 외부 데이터 SROIE+CORD-v2 (+0.74%p)
  6. K-Fold Fold3 선택 (+0.09%p)

### 미시도 전략 개요
**총 25개 전략 식별** (기존 21개 + 신규 4개), 5개 우선순위 그룹으로 분류:
- 🔥 **Tier 1 (즉시 실행)**: 4개 전략, 예상 +0.6~1.2%p ⭐ **신규 1개 추가**
- ⭐ **Tier 2 (고효과 중기)**: 8개 전략, 예상 +0.8~2.0%p ⭐ **신규 3개 추가**
- 💡 **Tier 3 (실험적 장기)**: 6개 전략, 예상 +0.3~0.7%p
- ⚠️ **Tier 4 (고위험)**: 4개 전략, 효과 불확실
- ❌ **Tier 5 (비추천)**: 3개 전략, 실패 가능성 높음

### 🆕 신규 추가 전략 (BREAKTHROUGH 잠재력)
1. **Tiny Box Loss 가중치 부여** (Tier 1) - 에러 케이스 직접 타겟, +0.4~0.7%p
2. **대규모 외부 데이터 Pre-training** (Tier 2) - 13,000장 통합, +0.8~1.2%p
3. **2단계 학습 파이프라인** (Tier 2) - Curriculum Learning, +0.3~0.6%p
4. **P2 Feature Pyramid 레벨 추가** (Tier 2) - 미세 텍스트 강화, +0.2~0.5%p

---

## 🎯 Part 1: Tier 1 전략 - 즉시 실행 가능 (0~2일)

### 1️⃣ Unclip Ratio 최적화 ⭐⭐⭐⭐⭐

**현재 상태**: 고정값 2.0 사용 (기본값)  
**문제점**: 모든 텍스트 크기에 동일한 확장 비율 적용  
**예상 개선**: +0.2~0.4%p | **소요 시간**: 10분 | **난이도**: ⭐

#### 배경 지식
```
Unclip Ratio: DBNet에서 축소된 텍스트 영역을 원래 크기로 확장하는 비율

과정:
  1. DBNet이 텍스트 중심부만 검출 (축소된 영역)
  2. Unclip으로 확장하여 전체 텍스트 커버
  3. 비율이 작으면: 텍스트 일부만 포함 (Recall ↓)
  4. 비율이 크면: 배경까지 포함 (Precision ↓)
```

#### 실행 전략
```yaml
# 탐색 범위
unclip_ratio_candidates: [1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15]

# Grid Search
for ratio in candidates:
    predictions = predict_with_ratio(val_set, ratio)
    hmean = evaluate(predictions, val_gt)
    if hmean > best_hmean:
        best_ratio = ratio
```

#### 예상 결과
```
최적 비율 발견 시:
  Before: H=98.63%, unclip_ratio=2.00
  After:  H=98.8~99.0%, unclip_ratio=1.95~2.05
  
에러 분석 기반 예상:
  - 작은 박스 (면적 <40px²): ratio ↑ (2.05~2.10)
  - 큰 박스 (면적 >5000px²): ratio ↓ (1.90~1.95)
```

#### 실행 명령어
```bash
cd /data/ephemeral/home/baseline_code

# Validation set Grid Search
python scripts/optimize_unclip_ratio.py \
  --checkpoint checkpoints/kfold/fold_3/fold3_best.ckpt \
  --val_json kfold_results/fold_3/val.json \
  --ratio_range 1.85 2.15 \
  --step 0.05
```

---

### 2️⃣ 🆕 Tiny Box Loss 가중치 부여 (Small Object 특화) ⭐⭐⭐⭐⭐

**현재 상태**: 모든 박스에 동일한 Loss 가중치 적용  
**문제점**: 20px² 최소 박스 검출 실패 (56_ERROR_ANALYSIS 확인)  
**예상 개선**: +0.4~0.7%p | **소요 시간**: 2시간 | **난이도**: ⭐⭐⭐

#### 에러 케이스 기반 필요성
```
56_ERROR_ANALYSIS 발견:
  소형 박스 카테고리 (20개 이미지):
    - 최소 박스 면적: 20px² (selectstar_000525.jpg)
    - 평균 최소 박스: 30px²
    - 검출 실패 위험: 매우 높음
  
현재 모델 특성:
  - 큰 박스 (>1000px²): Recall 99.5%
  - 작은 박스 (<100px²): Recall 95~97% (추정)
  → 작은 박스가 성능 병목
```

#### Focal Loss 변형 전략
```python
# 박스 크기별 Loss 가중치
def get_loss_weight(box_area):
    if box_area < 50:
        return 10.0   # 초소형 (극한 가중치)
    elif box_area < 100:
        return 5.0    # 소형
    elif box_area < 200:
        return 2.0    # 중소형
    else:
        return 1.0    # 표준

# DBNet Loss 수정
class DBLossWeighted(nn.Module):
    def forward(self, pred, gt, boxes):
        weights = torch.tensor([get_loss_weight(box.area()) for box in boxes])
        
        # Probability Map Loss (가중치 적용)
        prob_loss = F.binary_cross_entropy(
            pred['probability_map'], 
            gt['probability_map'],
            weight=weights  # 작은 박스 영역에 높은 가중치
        )
        
        # Threshold Map Loss (동일 가중치)
        thresh_loss = ...
        
        return prob_loss + thresh_loss
```

#### 기대 효과
```
작은 박스 검출 개선:
  Before: 100px² 이하 Recall 95%
  After:  100px² 이하 Recall 98~99%
  
전체 성능:
  소형 박스 비율: 전체의 약 15~20%
  개선 기여: Recall +0.4~0.6%p
  H-Mean: +0.4~0.7%p
```

#### 실행 방법
```python
# ocr/models/loss/db_loss_weighted.py 생성
class DBLossWeighted(DBLoss):
    def __init__(self, alpha=5.0, beta=10.0, negative_ratio=3.0):
        super().__init__(alpha, beta, negative_ratio)
        self.tiny_threshold = 50   # 초소형 기준
        self.small_threshold = 100 # 소형 기준
        self.tiny_weight = 10.0
        self.small_weight = 5.0
    
    def compute_area_weights(self, gt_boxes):
        areas = [box['area'] for box in gt_boxes]
        weights = []
        for area in areas:
            if area < self.tiny_threshold:
                weights.append(self.tiny_weight)
            elif area < self.small_threshold:
                weights.append(self.small_weight)
            else:
                weights.append(1.0)
        return torch.tensor(weights)

# configs/preset/models/loss/db_loss_weighted.yaml
loss:
  name: DBLossWeighted
  alpha: 5.0
  beta: 10.0
  tiny_threshold: 50
  small_threshold: 100
  tiny_weight: 10.0
  small_weight: 5.0
```

#### 실행 명령어
```bash
cd /data/ephemeral/home/baseline_code

# Loss 클래스 구현 (1시간)
# 위 코드를 ocr/models/loss/db_loss_weighted.py에 작성

# 재훈련 (1시간, Fold 3만)
python runners/train.py \
    preset=hrnet_w44_1024_external_weighted_loss \
    model.loss.name=DBLossWeighted \
    model.loss.tiny_weight=10.0 \
    model.loss.small_weight=5.0 \
    ++datasets.train_dataset.annotation_path=train_augmented_full.json \
    ++datasets.val_dataset.annotation_path=kfold_results/fold_3/val.json \
    exp_name=hrnet_w44_tiny_box_weighted \
    trainer.max_epochs=10  # Fine-tuning
```

---

### 3️⃣ WildReceipt 외부 데이터 추가 ⭐⭐⭐⭐

**현재 상태**: SROIE (626장) + CORD-v2 (800장) 사용, WildReceipt 미사용  
**예상 개선**: +0.3~0.5%p | **소요 시간**: 3시간 | **난이도**: ⭐⭐

#### 데이터 특성 비교
```
현재 데이터:
  기본 데이터: 3,272장 (100%)
  + SROIE:      626장 (+19.1%, 빽빽한 영수증)
  + CORD-v2:    800장 (+24.4%, 한글 복잡 레이아웃)
  ──────────────────────────────
  총계:       4,698장 (+43.6%)

추가 가능:
  + WildReceipt: 1,300장 (+39.7%, 구겨진/휘어진 영수증)
  ──────────────────────────────
  최종:         5,998장 (+83.3%)
```

#### 기대 효과
```
SROIE+CORD 기여도: +0.71%p (from 97.8% → 98.51%)

WildReceipt 추가 시:
  - 구겨진 영수증 대응력 향상 (에러 케이스 대응)
  - 극단적 종횡비 텍스트 처리 개선
  - 예상 기여: +0.3~0.5%p
```

#### 실행 명령어
```bash
# 1. WildReceipt 다운로드 (5분)
cd /data/ephemeral/home/data/pseudo_label
git clone https://github.com/clovaai/wildreceipt.git

# 2. 포맷 변환 (10분)
cd /data/ephemeral/home/baseline_code
python scripts/convert_wildreceipt.py \
  --input /data/ephemeral/home/data/pseudo_label/wildreceipt \
  --output /data/ephemeral/home/data/datasets/wildreceipt_converted.json

# 3. 데이터 병합 (5분)
python scripts/merge_datasets.py \
  --inputs train_augmented_full.json wildreceipt_converted.json \
  --output train_augmented_wildreceipt.json

# 4. 재훈련 (2.5시간)
python runners/train.py \
    preset=hrnet_w44_1024_external \
    ++datasets.train_dataset.annotation_path=train_augmented_wildreceipt.json \
    exp_name=hrnet_w44_wildreceipt \
    trainer.max_epochs=18
```

---

### 4️⃣ 후처리 초미세 조정 (0.215 → 0.210~0.220 범위) ⭐⭐⭐⭐

**현재 상태**: thresh=0.215, box_thresh=0.415 (48_comprehensive 최적값)  
**예상 개선**: +0.05~0.15%p | **소요 시간**: 5분 | **난이도**: ⭐

#### 에러 분석 기반 조정
```
56_ERROR_ANALYSIS_REPORT.md 발견사항:
  - 고밀도 이미지 (538 boxes/Mpx): Recall 손실 위험
  - 소형 박스 (20px² 최소): 검출 실패 가능성
  
→ Thresh를 소폭 낮춰 Recall 개선 시도
```

#### 탐색 공간
```yaml
# 기존 최적값 주변 세밀 탐색
thresh:     [0.210, 0.212, 0.215, 0.218, 0.220]
box_thresh: [0.410, 0.412, 0.415, 0.418, 0.420]

조합: 5×5 = 25개
소요 시간: 5분 (Validation set만 예측)
```

#### 예상 결과
```
99_comprehensive에서 관찰된 패턴:
  thresh vs Recall: 非단조 곡선
  0.215가 로컬 최대값이었으나, 외부 데이터 추가 후 최적점 이동 가능

예상 최적점:
  thresh: 0.212~0.218
  box_thresh: 0.415 (고정 or 0.410~0.420)
  H-Mean: 98.68~98.78%
```

#### 실행 명령어
```bash
python scripts/postprocess_grid_search.py \
  --checkpoint checkpoints/kfold/fold_3/fold3_best.ckpt \
  --thresh_range 0.210 0.220 --thresh_step 0.002 \
  --box_thresh_range 0.410 0.420 --box_thresh_step 0.005 \
  --output outputs/postprocess_fine_tuning
```

---

## ⭐ Part 2: Tier 2 전략 - 고효과 중기 (3~7일)

### 5️⃣ 🆕 대규모 외부 데이터 통합 Pre-training (13,000장) ⭐⭐⭐⭐⭐

**현재 상태**: SROIE (626장) + CORD-v2 (800장) = 4,698장  
**예상 개선**: +0.8~1.2%p | **소요 시간**: 8시간 | **난이도**: ⭐⭐⭐

#### 확장 데이터셋 구성
```
현재 데이터:
  대회 데이터:  3,272장 (100%)
  + SROIE:       626장 (+19.1%, 빽빽한 영수증)
  + CORD-v2:     800장 (+24.4%, 한글 복잡 레이아웃)
  ──────────────────────────────
  소계:        4,698장 (+43.6%)

추가 가능 데이터셋:
  + WildReceipt:  1,300장 (+39.7%, 구겨진/휘어진)
  + ICDAR 2019:   1,000장 (+30.6%, 다국어)
  + RVL-CDIP:     2,500장 (+76.4%, 문서 다양성)
  + SynthText:    3,500장 (+107%, 합성 데이터)
  ──────────────────────────────
  총계:        ~13,000장 (+297% 증가!) ⭐⭐⭐
```

#### 각 데이터셋의 역할
```
SROIE (626장):
  특화: 초밀집 텍스트 (개미 잡기)
  기여: 고밀도 이미지 대응
  
CORD-v2 (800장):
  특화: 한국어 복잡 레이아웃
  기여: 도메인 일치성
  
WildReceipt (1,300장):
  특화: 구겨진, 휘어진 영수증 (뱀 잡기)
  기여: 변형 강건성
  
ICDAR 2019 (1,000장):
  특화: 다국어, 다양한 폰트
  기여: 일반화 능력
  
RVL-CDIP (2,500장):
  특화: 문서 레이아웃 다양성
  기여: 구조적 이해력
  
SynthText (3,500장):
  특화: 합성 텍스트 (무한 생성 가능)
  기여: 데이터 증강 효과
```

#### 99_comprehensive 데이터 효과 검증
```
외부 데이터 기여도 분석:
  기본 → +626장 SROIE:     +0.35%p (추정)
  기본 → +800장 CORD:      +0.36%p (추정)
  기본 → +1,426장 총계:    +0.71%p
  
단위당 효과: +0.71%p / 1,426장 = 0.50%p per 1,000장

13,000장 적용 시:
  추가 데이터: 13,000 - 4,698 = 8,302장
  예상 기여: 0.50%p × 8.3 = +0.8~1.2%p ⭐⭐⭐
```

#### 실행 전략
```bash
# 1. 모든 외부 데이터셋 다운로드 (2시간)
cd /data/ephemeral/home/data/pseudo_label

# SROIE (이미 있음)
# CORD-v2 (이미 있음)

# WildReceipt
git clone https://github.com/clovaai/wildreceipt.git

# ICDAR 2019 RobustReading
wget https://rrc.cvc.uab.es/downloads/icdar2019_task1.zip
unzip icdar2019_task1.zip

# RVL-CDIP (subset)
# Hugging Face에서 다운로드
from huggingface_hub import snapshot_download
snapshot_download("aharley/rvl_cdip", repo_type="dataset", local_dir="./rvl_cdip")

# 2. 통합 포맷 변환 (3시간)
cd /data/ephemeral/home/baseline_code
python scripts/convert_all_external_datasets.py \
  --sroie ../data/pseudo_label/sroie \
  --cord ../data/pseudo_label/cord-v2 \
  --wildreceipt ../data/pseudo_label/wildreceipt \
  --icdar ../data/pseudo_label/icdar2019_task1 \
  --rvl_cdip ../data/pseudo_label/rvl_cdip \
  --output ../data/datasets/external_unified_13k.json

# 3. 대회 데이터와 병합 (10분)
python scripts/merge_datasets.py \
  --base train_augmented_full.json \
  --external external_unified_13k.json \
  --output train_mega_dataset_13k.json \
  --validate  # 중복 제거, 품질 검증

# 4. Pre-training (3시간)
python runners/train.py \
    preset=hrnet_w44_1024_pretrain \
    ++datasets.train_dataset.annotation_path=train_mega_dataset_13k.json \
    exp_name=hrnet_w44_pretrain_13k \
    trainer.max_epochs=15
```

#### 예상 결과
```
Pre-training 효과:
  데이터 다양성: +297% → 일반화 능력 극대화
  도메인 커버리지:
    - 밀집 텍스트 (SROIE)
    - 한국어 (CORD)
    - 변형 (WildReceipt)
    - 다국어 (ICDAR)
    - 다양한 레이아웃 (RVL-CDIP)
  
성능 예상:
  Validation: H-Mean 99.0~99.3%
  Test/Leaderboard: H-Mean 99.2~99.5%
```

---

### 6️⃣ 🆕 2단계 학습 파이프라인 (Curriculum Learning) ⭐⭐⭐⭐

**현재 상태**: 단일 해상도 (1024px) 단일 데이터셋 학습  
**예상 개선**: +0.3~0.6%p | **소요 시간**: 6시간 | **난이도**: ⭐⭐⭐

#### Curriculum Learning 전략
```
Stage 1: Pre-training (외부 데이터 포함)
  목적: 넓은 일반화 능력 확보
  데이터: 13,000장 통합 데이터셋
  해상도: 1024px (효율적 학습)
  Epochs: 15
  학습률: 0.001
  
Stage 2: Fine-tuning (대회 데이터만)
  목적: 대회 특화 정밀 최적화
  데이터: 3,272장 대회 데이터만
  해상도: 1280px (고해상도 정밀)
  Epochs: 8
  학습률: 0.0001 (1/10로 감소)
```

#### 99_comprehensive 해상도 분석 재검토
```
기존 발견:
  1024px → 1280px: +0.0~0.1%p (효과 거의 없음)
  
하지만:
  "1280px 단독 학습"과 "1024 Pre-train → 1280 Fine-tune"은 다름!
  
이유:
  1. Transfer Learning 효과
     - 1024px에서 충분한 feature 학습
     - 1280px에서 미세 조정만 수행
     
  2. 고해상도의 정확한 역할
     - 1280px 처음부터: 수렴 느림, 과적합 위험
     - 1280px Fine-tune: 정밀도만 개선
     
  3. 대회 데이터 집중
     - Stage 2에서 외부 데이터 배제
     - 대회 특성에 최적화
```

#### 실행 방법
```bash
cd /data/ephemeral/home/baseline_code

# Stage 1: Pre-training @ 1024px (3시간)
python runners/train.py \
    preset=hrnet_w44_1024_stage1 \
    ++datasets.train_dataset.annotation_path=train_mega_dataset_13k.json \
    ++datasets.val_dataset.annotation_path=kfold_results/fold_3/val.json \
    datasets.image_size=1024 \
    optimizer.lr=0.001 \
    trainer.max_epochs=15 \
    exp_name=stage1_pretrain_1024px_13k

# Stage 2: Fine-tuning @ 1280px (3시간)
python runners/train.py \
    preset=hrnet_w44_1280_stage2 \
    ++resume_from=outputs/stage1_pretrain_1024px_13k/checkpoints/last.ckpt \
    ++datasets.train_dataset.annotation_path=train.json \
    ++datasets.val_dataset.annotation_path=kfold_results/fold_3/val.json \
    datasets.image_size=1280 \
    optimizer.lr=0.0001 \
    trainer.max_epochs=8 \
    exp_name=stage2_finetune_1280px_competition
```

#### 기대 효과
```
Stage 1 (Pre-training):
  넓은 일반화 능력 확보
  다양한 텍스트 패턴 학습
  
Stage 2 (Fine-tuning):
  대회 데이터 특화:
    - 한국어 영수증 레이아웃
    - 특정 폰트, 형식
  고해상도 정밀도:
    - 작은 글자 경계선 개선
    - 긴 텍스트 라인 정확도 향상
  
최종 예상:
  Stage 1 단독: H-Mean 99.0~99.2%
  Stage 2 추가: H-Mean 99.3~99.6%
  순수 개선: +0.3~0.6%p
```

---

### 7️⃣ 🆕 P2 Feature Pyramid 레벨 추가 (High-Res FPN) ⭐⭐⭐⭐

**현재 상태**: FPN P3~P7 레벨 사용 (1/8 ~ 1/128 해상도)  
**예상 개선**: +0.2~0.5%p | **소요 시간**: 5시간 | **난이도**: ⭐⭐⭐⭐

#### Feature Pyramid Network 구조
```
현재 FPN (DBNet 기본):
  
  Input Image (1024×1024)
    ↓
  Encoder (HRNet-W44)
    ├─ 1/4:  256×256  (C2, 미사용)
    ├─ 1/8:  128×128  (C3) → P3 ⭐
    ├─ 1/16:  64×64   (C4) → P4
    ├─ 1/32:  32×32   (C5) → P5
    └─ 1/64:  16×16   (C6) → P6
  
  FPN Neck:
    P3, P4, P5, P6 → Lateral + Top-down
  
  Head:
    Fused features → Probability Map

문제점:
  P3 (1/8 해상도)가 가장 높은 해상도
  → 작은 텍스트 (20px²)는 1/8로 줄면 2.5px²
  → 정보 소실 위험
```

#### P2 레벨 추가 설계
```
개선된 FPN:
  
  Input Image (1024×1024)
    ↓
  Encoder (HRNet-W44)
    ├─ 1/4:  256×256  (C2) → P2 ⭐⭐⭐ 신규 추가
    ├─ 1/8:  128×128  (C3) → P3
    ├─ 1/16:  64×64   (C4) → P4
    ├─ 1/32:  32×32   (C5) → P5
    └─ 1/64:  16×16   (C6) → P6
  
  FPN Neck:
    P2, P3, P4, P5, P6 → Enhanced fusion
  
  Head:
    P2 주도 (작은 객체) + P3~P6 보조

장점:
  작은 텍스트 (20px²):
    - 1/8 해상도: 2.5px² (정보 부족)
    - 1/4 해상도: 5px² (충분한 정보)
  
  경계선 정밀도:
    - 1/4 해상도로 복원 → 더 정밀한 polygon
```

#### 구현 방법
```python
# ocr/models/decoder/fpn_with_p2.py
class FPNDecoderWithP2(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=256):
        super().__init__()
        
        # Lateral connections (C2부터 시작)
        self.lateral_c2 = nn.Conv2d(in_channels[0], out_channels, 1)  # 신규
        self.lateral_c3 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels[3], out_channels, 1)
        
        # Top-down pathway
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)  # 신규
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, features):
        c2, c3, c4, c5 = features  # HRNet의 4개 stage
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2)  # 신규
        
        # Smooth
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)  # 신규
        
        # Upsample all to P2 resolution for fusion
        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear')
        p4_up = F.interpolate(p4, scale_factor=4, mode='bilinear')
        p5_up = F.interpolate(p5, scale_factor=8, mode='bilinear')
        
        # Weighted fusion (P2 주도)
        fused = 0.5 * p2 + 0.25 * p3_up + 0.15 * p4_up + 0.10 * p5_up
        
        return fused

# configs/preset/models/decoder/fpn_with_p2.yaml
decoder:
  name: FPNDecoderWithP2
  in_channels: [64, 128, 256, 512]  # HRNet-W44 outputs
  out_channels: 256
  use_p2: true
```

#### 주의사항
```
메모리 증가:
  P2 (1/4 해상도) = P3의 4배 메모리
  → 배치 크기 4 → 2로 감소 필요
  
학습 시간:
  FPN 연산 증가: +30% 시간
  3시간 → 4시간 예상
  
하지만:
  작은 텍스트 검출 개선이 목표라면 투자 가치 있음
```

#### 실행 명령어
```bash
cd /data/ephemeral/home/baseline_code

# FPN P2 구현 (1시간)
# 위 코드를 ocr/models/decoder/fpn_with_p2.py에 작성

# 학습 (4시간)
python runners/train.py \
    preset=hrnet_w44_1024_fpn_p2 \
    model.decoder.name=FPNDecoderWithP2 \
    model.decoder.use_p2=true \
    ++datasets.train_dataset.annotation_path=train_augmented_full.json \
    ++datasets.val_dataset.annotation_path=kfold_results/fold_3/val.json \
    dataloaders.train_dataloader.batch_size=2 \
    trainer.max_epochs=12 \
    exp_name=hrnet_w44_fpn_p2_small_objects
```

#### 예상 결과
```
소형 박스 (20~100px²) 검출:
  Before P2: Recall 95~97%
  After P2:  Recall 98~99%
  
경계선 정밀도:
  Polygon points: 더 정확한 좌표
  IoU with GT: +1~2% 향상
  
전체 성능:
  Recall: +0.2~0.4%p
  Precision: +0.1~0.2%p (경계선 정밀도)
  H-Mean: +0.2~0.5%p
```

---

### 8️⃣ EfficientNet-B3/B4 백본 실험 ⭐⭐⭐⭐⭐

**현재 상태**: HRNet-W44 사용  
**예상 개선**: +0.5~0.8%p (단일 모델) 또는 혼합 앙상블로 +0.8~1.2%p  
**소요 시간**: 3시간 (학습) | **난이도**: ⭐⭐

#### EfficientNet의 강점
```
HRNet-W44 vs EfficientNet-B4:

HRNet-W44:
  장점: 고해상도 유지, 텍스트 경계선 정밀
  단점: Parameters 67.8M (무거움)
  특화: 텍스트 검출

EfficientNet-B4:
  장점: 
    - Compound Scaling (depth+width+resolution 동시)
    - ImageNet 82.9% Top-1 (HRNet보다 높음)
    - 19.3M parameters (가벼움)
  특화: 일반 물체 검출, Transfer Learning 우수
```

#### 전략 옵션

**옵션 A: 단독 실험**
```bash
python runners/train.py \
    preset=efficientnet_b4_1024 \
    model.encoder.model_name=tf_efficientnet_b4 \
    datasets.image_size=1024 \
    trainer.max_epochs=18 \
    exp_name=efficientnet_b4_external_data
```

**옵션 B: 혼합 앙상블 (추천)**
```python
# HRNet + EfficientNet 백본 다양성 활용
models = [
    'hrnet_w44_fold3.ckpt',      # H-Mean 98.63%
    'efficientnet_b4_best.ckpt', # 예상 98.5~98.7%
]

# Weighted Box Fusion (IoU 0.7)
ensemble_result = wbf(models, weights=[0.55, 0.45], iou_thr=0.7)
```

#### 예상 결과
```
단일 모델:
  EfficientNet-B4: H=98.5~98.7% (HRNet보다 약간 낮을 가능성)
  
혼합 앙상블:
  HRNet (텍스트 경계선 강점) + EfficientNet (일반화 강점)
  → 상호 보완 효과
  예상: H=98.9~99.1% (+0.3~0.5%p)
```

---

### 9️⃣ Multi-Scale Test-Time Augmentation ⭐⭐⭐⭐

**현재 상태**: 단일 스케일 (1024px) 예측, 08_tta_failure로 HFlip TTA 실패  
**예상 개선**: +0.2~0.4%p | **소요 시간**: 30분 | **난이도**: ⭐⭐⭐

#### 08_tta_failure 원인 분석
```
실패 원인:
  ✗ HorizontalFlip 좌표 변환 미구현
  ✗ 잘못된 위치의 박스 215개 생성
  ✗ Precision 25.5% 폭락

교훈:
  → Flip TTA는 구현 복잡도 높음
  → Multi-Scale TTA가 더 안전하고 효과적
```

#### Multi-Scale TTA 전략
```python
# 3가지 스케일로 예측
scales = [960, 1024, 1088]  # ±6.25% 범위

predictions = []
for scale in scales:
    resized_image = resize(image, scale)
    pred = model(resized_image)
    pred_original = rescale_boxes(pred, original_size)
    predictions.append(pred_original)

# Weighted Box Fusion
final = wbf(
    predictions,
    weights=[0.25, 0.50, 0.25],  # 1024px에 가중치
    iou_thr=0.6
)
```

#### 기대 효과
```
Small Boxes (에러 분석 카테고리):
  960px:  작은 글자 일부 누락 가능
  1024px: 현재 최적
  1088px: 작은 글자 추가 검출 (+6.25% 해상도)
  
  → Fusion으로 누락 보완, Recall +0.2~0.3%p 예상
```

#### 실행 명령어
```bash
python scripts/predict_multiscale_tta.py \
  --checkpoint checkpoints/kfold/fold_3/fold3_best.ckpt \
  --scales 960 1024 1088 \
  --weights 0.25 0.50 0.25 \
  --iou_threshold 0.6 \
  --output outputs/multiscale_tta
```

---

### 🔟 Anchor Box 종횡비 추가/조정 ⭐⭐⭐⭐

**현재 상태**: DBNet 기본 설정 (비율 미조정)  
**예상 개선**: +0.1~0.3%p | **소요 시간**: 4시간 (재학습) | **난이도**: ⭐⭐⭐

#### 에러 분석 기반 필요성
```
56_ERROR_ANALYSIS 발견:
  극단적 종횡비 (AR > 6.0):
    - selectstar_000827.jpg: AR=7.68 (최대)
    - 20개 이미지에서 AR>6.0
  
  → 기본 Anchor Box로 커버 부족 가능성
```

#### DBNet Anchor Box 설정
```yaml
# 현재 (추정, 기본값)
anchor_ratios:
  - 0.2   # 1:5 (매우 가로로 긴)
  - 0.5   # 1:2
  - 1.0   # 정사각형
  - 2.0   # 2:1
  - 5.0   # 5:1

# 추가 제안
anchor_ratios:
  - 0.13  # 1:7.5 (AR=7.68 대응) ⭐
  - 0.2
  - 0.5
  - 1.0
  - 2.0
  - 5.0
  - 7.5   # 극단 비율 추가 ⭐
```

#### 실행 방법
```yaml
# configs/preset/models/model_hrnet_w44_custom_anchors.yaml
model:
  architecture:
    encoder:
      model_name: hrnet_w44
    decoder:
      name: FPNDecoder
    head:
      name: DBHead
      anchor_ratios: [0.13, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5]  # 수정
```

#### 예상 결과
```
극단 종횡비 이미지 (20개):
  Before: AR>6.0 텍스트 검출률 ~95%
  After:  AR>6.0 텍스트 검출률 ~98% (+3%p)
  
전체 성능:
  Recall: +0.1~0.2%p
  H-Mean: +0.1~0.3%p
```

---

### 1️⃣1️⃣ Deformable Convolution (DCN) 적용 ⭐⭐⭐

**현재 상태**: 일반 Convolution 사용  
**예상 개선**: +0.3~0.5%p | **소요 시간**: 5시간 (구현+학습) | **난이도**: ⭐⭐⭐⭐

#### Deformable Convolution 장점
```
일반 Convolution:
  3×3 kernel → 고정된 9개 위치 샘플링
  문제: 구겨진, 휘어진 텍스트에 부적합

Deformable Convolution:
  각 샘플링 위치가 학습 가능한 offset
  → 텍스트 형태에 맞춰 동적 샘플링
  
효과:
  ✓ 구겨진 영수증 대응 (WildReceipt 추가 시 시너지)
  ✓ 극단 종횡비 텍스트 처리
  ✓ 경계선 정밀도 향상
```

#### 구현 방법
```python
# HRNet Backbone에 DCN 적용
from torchvision.ops import DeformConv2d

class HRNetWithDCN(nn.Module):
    def __init__(self):
        # Stage 3, 4의 convolution을 DCN으로 교체
        self.stage3_dcn = DeformConv2d(256, 256, 3, padding=1)
        self.stage4_dcn = DeformConv2d(512, 512, 3, padding=1)
```

#### 예상 결과
```
CVPR 2017 논문 결과:
  일반 Conv → DCN: COCO Detection +5~10% mAP
  
OCR 적용 예상:
  Recall: +0.3~0.4%p (구겨진 텍스트)
  Precision: 유지 or +0.1%p
  H-Mean: +0.3~0.5%p
```

---

### 1️⃣2️⃣ Vision Transformer (ViT) 또는 Swin Transformer 백본 ⭐⭐⭐⭐

**현재 상태**: CNN 기반 (HRNet-W44)  
**예상 개선**: +0.5~1.0%p | **소요 시간**: 6시간 | **난이도**: ⭐⭐⭐⭐

#### Transformer 장점
```
CNN (HRNet):
  장점: 지역 특징 추출 강력, 고해상도 유지
  단점: 전역 문맥 부족, Receptive field 제한

Vision Transformer:
  장점:
    - 전역 Self-Attention (이미지 전체 문맥)
    - Long-range dependency 모델링
    - 텍스트 블록 간 관계 파악
  단점:
    - 많은 데이터 필요 (외부 데이터 추가로 해결)
    - 학습 시간 증가
```

#### 추천 모델
```
1. Swin Transformer-B (추천 ⭐⭐⭐⭐⭐)
   - Hierarchical structure (HRNet과 유사)
   - Window-based attention (효율적)
   - ImageNet-22K pretrained 가능
   
2. ViT-Base/16
   - 표준 Transformer
   - Pretrained 모델 풍부
```

#### 실행 전략
```yaml
# Swin Transformer 적용
model:
  encoder:
    model_name: swin_base_patch4_window7_224
    pretrained: true
    in_chans: 3
  decoder:
    name: FPNDecoder
  head:
    name: DBHead
```

#### 예상 결과
```
문헌 조사 결과:
  CNN → Transformer: +1~3% 일반적
  
외부 데이터 5,998장 기반:
  데이터 충분성: 충족 (Transformer는 >5,000장 권장)
  예상 H-Mean: 98.8~99.3%
```

---

## 💡 Part 3: Tier 3 전략 - 실험적 장기 (7~14일)

### 1️⃣3️⃣ Pseudo-Labeling (Self-Training) ⭐⭐⭐

**예상 개선**: +0.3~0.6%p | **소요 시간**: 8시간 | **난이도**: ⭐⭐⭐⭐

#### 전략
```
1. 현재 최고 모델 (H=98.63%)로 Test set 예측
2. 고신뢰도 예측만 선별 (confidence > 0.95)
3. Pseudo-GT로 활용하여 재훈련
4. 반복 (2~3 iteration)
```

#### 기대 효과
```
Test set 특성 학습:
  - Train/Test distribution gap 감소
  - 일반화 성능 향상
  
예상:
  1st iteration: +0.2~0.3%p
  2nd iteration: +0.1~0.2%p
  Total: +0.3~0.5%p
```

---

### 1️⃣4️⃣ FP16 Mixed Precision + Batch Size 증가 ⭐⭐⭐

**예상 개선**: +0.1~0.2%p | **소요 시간**: 2시간 | **난이도**: ⭐⭐

#### 전략
```yaml
# 현재
trainer:
  precision: 32
  batch_size: 4

# 변경
trainer:
  precision: 16
  amp_backend: native
  batch_size: 8  # or 12
```

#### 기대 효과
```
Batch Size 증가:
  ✓ 안정적인 Gradient (노이즈 감소)
  ✓ BatchNorm 통계 정확도 향상
  
예상:
  H-Mean: +0.1~0.2%p
  학습 시간: 1.3배 단축
```

---

### 1️⃣5️⃣ ConvNeXt 백본 실험 ⭐⭐⭐

**예상 개선**: +0.4~0.7%p | **소요 시간**: 4시간 | **난이도**: ⭐⭐⭐

#### ConvNeXt 특징
```
"Transformer를 이긴 CNN" (CVPR 2022)
  - ResNet 디자인 개선
  - Transformer의 강점 통합
  - HRNet보다 효율적
```

---

### 1️⃣6️⃣ Focal Loss 또는 DIoU Loss 적용 ⭐⭐

**예상 개선**: +0.1~0.3%p | **소요 시간**: 3시간 | **난이도**: ⭐⭐⭐

#### Loss Function 변경
```python
# 현재: L1 Loss (DBNet 기본)
# 변경: DIoU Loss (경계선 정밀도 향상)

class DIoULoss(nn.Module):
    """Distance-IoU Loss for better bbox regression"""
    pass
```

---

### 1️⃣7️⃣ CutOut/GridMask Augmentation ⭐⭐

**예상 개선**: +0.1~0.2%p | **소요 시간**: 2시간 | **난이도**: ⭐⭐

#### Augmentation 강화
```yaml
augmentation:
  - GridMask:      # 격자 무늬 마스킹
      ratio: 0.6
  - CutOut:        # 랜덤 영역 제거
      num_holes: 3
```

---

### 1️⃣8️⃣ Knowledge Distillation (Teacher-Student) ⭐⭐⭐

**예상 개선**: +0.2~0.4%p | **소요 시간**: 8시간 | **난이도**: ⭐⭐⭐⭐

#### 전략
```
Teacher: HRNet-W44 (98.63%)
Student: EfficientNet-B3 (가벼운 모델)

Distillation:
  Feature-level: Stage별 feature map matching
  Response-level: Soft labels 학습
  
목표:
  Student가 Teacher 성능 근접 + 추론 속도 2배
```

---

## ⚠️ Part 4: Tier 4 전략 - 고위험 (효과 불확실)

### 1️⃣9️⃣ 1536px 초고해상도 학습

**위험**: 99_comprehensive 발견 - 1280px 이상은 효과 거의 없음  
**예상**: +0.0~0.1%p (비효율)

---

### 2️⃣0️⃣ Soft Voting with Rescue Mechanism

**위험**: 53_ensemble_failure - 이미 우수한 모델(98%) 앙상블은 역효과  
**예상**: -0.2~+0.1%p (불확실)

---

### 2️⃣1️⃣ NMS/WBF 파라미터 재조정 앙상블

**위험**: IoU threshold 조정으로 53번 실패 극복 시도  
**예상**: -0.5~+0.2%p (위험)

---

### 2️⃣2️⃣ Learning Rate Warmup + Cosine Annealing 재조정

**위험**: 이미 Grid Search로 최적화됨  
**예상**: +0.0~0.05%p (미미)

---

## ❌ Part 5: Tier 5 전략 - 비추천 (실패 가능성 높음)

### 2️⃣3️⃣ HorizontalFlip TTA

**실패 사례**: 08_tta_failure_analysis.md  
**결과**: H-Mean 18.7% 폭락  
**이유**: 좌표 변환 복잡성, 구현 오류 위험

---

### 2️⃣4️⃣ K-Fold 5-Fold Voting Ensemble

**실패 사례**: 53_ensemble_failure_analysis.md  
**결과**: H-Mean 9.76%p 하락 (98.63% → 88.87%)  
**이유**: 이미 98% 이상 모델은 앙상블 역효과

---

### 2️⃣5️⃣ ResNet101/ResNet152 백본

**이유**: 99_comprehensive - HRNet이 이미 ResNet 계열 초월  
**예상**: 0%p 또는 마이너스

---

## 🎯 Part 6: 실행 우선순위 로드맵

### Phase 1: Quick Wins (1주일, 예상 +0.8~1.5%p) 🔥

```
Day 1:
  ✅ Unclip Ratio 최적화 (10분)           → +0.2~0.4%p
  ✅ 후처리 초미세 조정 (5분)             → +0.05~0.15%p
  ⭐ Tiny Box Loss 가중치 구현 (2시간)   → +0.4~0.7%p  🆕
  
예상 누적: 98.63% → 99.25~99.88%

Day 2-3:
  ✅ WildReceipt + 추가 외부 데이터 (5시간)
     - WildReceipt (1,300장)
     - ICDAR 2019 (1,000장)                 → +0.3~0.5%p
  
예상 누적: 99.25% → 99.55~100%

실행 순서:
  1. Unclip ratio grid search → 최적값 확정
  2. Tiny Box Loss 구현 및 Fine-tuning (1시간)
  3. 후처리 thresh/box_thresh 미세조정
  4. WildReceipt + 추가 데이터 준비
  5. 재훈련 (overnight 3시간)
```

### Phase 2: 고효과 실험 (2주일, 예상 +1.0~2.0%p 추가) ⭐

```
Week 2:
  🆕 대규모 Pre-training (8시간)          → +0.8~1.2%p
     - 13,000장 통합 데이터셋
     - SROIE + CORD + WildReceipt + ICDAR + RVL-CDIP
  
  🆕 2단계 학습 파이프라인 (6시간)    → +0.3~0.6%p
     - Stage 1: 1024px Pre-training (3시간)
     - Stage 2: 1280px Fine-tuning (3시간)
  
  🆕 P2 Feature Pyramid (5시간)          → +0.2~0.5%p
     - FPN에 고해상도 레벨 추가
     - 소형 박스 특화
  
  ⭐ Multi-Scale TTA (30분)               → +0.3%p
  ⭐ EfficientNet-B4 학습 (3시간)
  ⭐ HRNet + EffNet 혼합 앙상블 (1시간)
  
예상 누적: 99.55% → 99.8~100%+
```

### Phase 3: 장기 혁신 (추가 2주, 예상 +0.5~1.0%p) 💡

```
Week 3-4:
  💡 Swin Transformer 학습 (6시간)
  💡 Pseudo-Labeling 2 iterations (8시간)
  💡 Deformable Convolution (5시간)
  
예상 최종: 99.8% → 99.9~100%+
```

---

## 📊 Part 7: 예상 성능 궤적

```
H-Mean 성능 변화 (예상)

100% ┤                                    ◆ 99.9% (Phase 3)
     │                                ┌───┘
 99% ┤                            ┌───┘ 99.5% (Phase 2)
     │                         ┌──┘
     │                      ┌──┘ 99.15% (Phase 1)
     │                   ┌──┘
 98% ┼───────────────────┘ 98.63% (현재)
     │
     └──────────────────────────────────────► 실행 단계
       현재    Phase 1   Phase 2   Phase 3
              Quick     고효과    장기
              Wins      실험     혁신
       
목표: 99.0% 돌파 (Phase 1-2로 달성 가능)
```

---

## 💰 Part 8: 비용 대비 효과 분석

| 전략 | GPU 시간 | 구현 난이도 | 예상 효과 | ROI (효과/시간) | 추천도 |
|------|---------|------------|----------|----------------|--------||
| **Unclip Ratio** | 0.2h | ⭐ | +0.3%p | 1.50 %p/h | ⭐⭐⭐⭐⭐ |
| **후처리 미세조정** | 0.1h | ⭐ | +0.1%p | 1.00 %p/h | ⭐⭐⭐⭐⭐ |
| **🆕 Tiny Box Loss** | 2h | ⭐⭐⭐ | +0.6%p | 0.30 %p/h | ⭐⭐⭐⭐⭐ |
| **WildReceipt** | 3h | ⭐⭐ | +0.4%p | 0.13 %p/h | ⭐⭐⭐⭐ |
| **🆕 13K Pre-train** | 8h | ⭐⭐⭐ | +1.0%p | 0.13 %p/h | ⭐⭐⭐⭐⭐ |
| **🆕 2단계 학습** | 6h | ⭐⭐⭐ | +0.5%p | 0.08 %p/h | ⭐⭐⭐⭐ |
| **🆕 P2 FPN** | 5h | ⭐⭐⭐⭐ | +0.4%p | 0.08 %p/h | ⭐⭐⭐⭐ |
| **Multi-Scale TTA** | 0.5h | ⭐⭐⭐ | +0.3%p | 0.60 %p/h | ⭐⭐⭐⭐⭐ |
| **EfficientNet-B4** | 3h | ⭐⭐ | +0.6%p | 0.20 %p/h | ⭐⭐⭐⭐ |
| **Swin Transformer** | 6h | ⭐⭐⭐⭐ | +0.7%p | 0.12 %p/h | ⭐⭐⭐ |
| **Pseudo-Label** | 8h | ⭐⭐⭐⭐ | +0.4%p | 0.05 %p/h | ⭐⭐ |

**최고 ROI Top 5** (🆕 = 신규 전략):
1. ⭐ Unclip Ratio 최적화: 1.50 %p/h
2. ⭐ 후처리 미세조정: 1.00 %p/h
3. ⭐ Multi-Scale TTA: 0.60 %p/h
4. 🆕 **Tiny Box Loss 가중치**: 0.30 %p/h  ← 신규, 고효과!
5. 🆕 **13K Pre-training**: 0.13 %p/h  ← 절대 효과 최대!

**주목**: Tiny Box Loss는 ROI도 높고 절대 효과(+0.6%p)도 크므로 1순위 권장!

---

## 🔬 Part 9: 56_ERROR_ANALYSIS 연계 최적화

### 에러 케이스별 대응 전략

#### 1. 고밀도 이미지 (170.7+ boxes/Mpx)
**문제**: 538 boxes/Mpx 최대, False Negative 위험  
**대응 전략**:
- ✅ Unclip Ratio ↑ (2.0 → 2.05~2.10)
- ✅ Thresh ↓ (0.215 → 0.210)
- ✅ Multi-Scale TTA (1088px 추가)

#### 2. 소형 박스 (면적 < 40px²)
**문제**: 20px² 최소, 작은 텍스트 검출 실패  
**대응 전략**:
- ✅ Multi-Scale TTA (1088px로 해상도 상승)
- ✅ Deformable Conv (작은 영역 adaptive sampling)
- ✅ Box Threshold ↓ (0.415 → 0.410)

#### 3. 극단적 종횡비 (AR > 6.0)
**문제**: AR=7.68 최대, Anchor box 매칭 실패  
**대응 전략**:
- ✅ Anchor Ratio 추가 (0.13, 7.5)
- ✅ Deformable Conv
- ✅ WildReceipt 데이터 (긴 텍스트 다수)

#### 4. 다수 박스 (155+ boxes/image)
**문제**: 276개 최대, NMS 복잡도  
**대응 전략**:
- ✅ Max Candidates ↑ (500 → 700)
- ✅ NMS IoU threshold 조정 (0.5 → 0.6)

---

## 📋 Part 10: 실행 체크리스트

### Phase 1 체크리스트 (1주일)

```
□ Day 1: Unclip Ratio 최적화
  □ Grid Search 스크립트 작성 (30분)
  □ Validation set 실행 (10분)
  □ 최적값 확정 및 config 업데이트 (5분)
  
□ Day 1: Tiny Box Loss 가중치 구현 🆕
  □ DBLossWeighted 클래스 구현 (1시간)
  □ Config 파일 설정 (30분)
  □ Fine-tuning 훈련 실행 (1시간)
  
□ Day 1: 후처리 미세조정
  □ Thresh/Box_Thresh Grid Search (5분)
  □ 최적 조합 선정 및 Test set 예측 (2분)
  
□ Day 2: 대규모 외부 데이터 준비 🆕
  □ WildReceipt + ICDAR 다운로드 (1시간)
  □ 포맷 변환 스크립트 작성 (1시간)
  □ 변환 실행 및 검증 (1시간)
  □ 데이터 병합 (30분)
  
□ Day 2-3: 재훈련
  □ 학습 시작 (overnight, 3시간)
  □ Validation 성능 확인
  □ Test set 예측 및 제출
```

### Phase 2 체크리스트 (2주차)

```
□ 13K Pre-training 🆕
  □ 모든 외부 데이터셋 다운로드 (2시간)
  □ 통합 포맷 변환 (3시간)
  □ Pre-training 학습 (3시간)
  
□ 2단계 학습 파이프라인 🆕
  □ Stage 1: 1024px Pre-training (3시간)
  □ Stage 2: 1280px Fine-tuning (3시간)
  □ 성능 비교 분석
  
□ P2 Feature Pyramid 구현 🆕
  □ FPNDecoderWithP2 클래스 구현 (1시간)
  □ Config 설정 및 테스트
  □ 훈련 실행 (4시간)
  
□ Multi-Scale TTA
  □ TTA 스크립트 작성 (1시간)
  □ 3-Scale 예측 실행 (30분)
  □ WBF 병합 및 제출 (5분)
  
□ EfficientNet-B4 학습
  □ Config 파일 생성
  □ 학습 실행 (3시간)
  □ Validation 평가
  
□ 혼합 앙상블
  □ HRNet + EffNet WBF 구현
  □ IoU threshold 실험 (0.6, 0.7, 0.8)
  □ 최고 성능 조합 제출
```

---

## 🎓 Part 11: 핵심 인사이트 및 주의사항

### ✅ DO: 실행 권장

1. **초기 빠른 개선에 집중**
   - Unclip ratio, 후처리 조정은 투자 대비 효과 최고
   
2. **데이터 다양성 확보**
   - WildReceipt 추가로 에러 케이스 직접 대응
   
3. **Multi-Scale TTA 활용**
   - HFlip보다 안전하고 효과적
   
4. **백본 다양성 실험**
   - EfficientNet, Swin Transformer 시도
   
5. **에러 분석 기반 최적화**
   - 56_ERROR_ANALYSIS의 4가지 카테고리 직접 대응

### ❌ DON'T: 실행 비추천

1. **Voting/WBF 앙상블 금지**
   - 98% 이상 모델은 앙상블 역효과 (53_ensemble_failure)
   
2. **HFlip TTA 구현 금지**
   - 좌표 변환 복잡성, 08_tta_failure 재발 위험
   
3. **1536px+ 초고해상도 금지**
   - 1280px 이상은 효과 없음 (99_comprehensive)
   
4. **ResNet 계열 추가 실험 금지**
   - HRNet이 이미 초월
   
5. **과도한 Grid Search**
   - 이미 최적화됨, 추가 효과 미미

---

## 📞 Part 12: 결론 및 권장 실행 경로

### 최종 권장 시나리오

**보수적 경로 (99.0% 목표, 1주일)**:
```
1. Unclip Ratio (10분)        → 98.83%
2. 후처리 미세조정 (5분)       → 98.88%
3. WildReceipt 재훈련 (3시간)  → 99.18%
──────────────────────────────
총 소요: 3.25시간
예상 결과: 99.0~99.2% ✅
```

**적극적 경로 (99.5% 목표, 2주일)**:
```
보수적 경로 (99.18%)
  + Multi-Scale TTA (30분)     → 99.38%
  + EfficientNet-B4 (3시간)    → 99.42%
  + 혼합 앙상블 (1시간)         → 99.58%
──────────────────────────────
총 소요: 7.75시간
예상 결과: 99.4~99.6% ✅✅
```

**혁신적 경로 (99.8%+ 목표, 4주일)**:
```
적극적 경로 (99.58%)
  + Swin Transformer (6시간)   → 99.68%
  + Pseudo-Labeling (8시간)    → 99.78%
  + Deformable Conv (5시간)    → 99.85%
──────────────────────────────
총 소요: 26.75시간 (약 3.5일 GPU 시간)
예상 결과: 99.7~99.9% ✅✅✅
```

### 최종 의사결정

**목표가 99.0% 달성이라면**:
→ **보수적 경로** 실행 (1주일, 성공 확률 95%)
   특히 **Tiny Box Loss**가 핵심! 🆕

**목표가 99.5%+ 극한 최적화라면**:
→ **적극적 경로** 실행 (2주일, 성공 확률 80%)
   13K Pre-training + 2단계 학습 + P2 FPN 조합 🆕

**연구/실험 목적이라면**:
→ **혁신적 경로** 실행 (4주일, 성공 확률 60%)

---

## 📚 참고 문헌

- `99_comprehensive_ocr_insights_report.md`: 6단계 모멘텀 분석
- `56_ERROR_ANALYSIS_REPORT.md`: 에러 케이스 식별
- `53_ensemble_failure_analysis_report.md`: 앙상블 실패 교훈
- `08_tta_failure_analysis.md`: TTA 구현 주의사항
- `16_Leaderboard score maximization`: EfficientNet 전략
- `42_SROIE_CORD_WildReceipt_GT_확보_가이드.md`: 외부 데이터 가이드

---

**작성자**: GitHub Copilot  
**작성일**: 2026-02-13  
**버전**: v2.0 (신규 전략 4개 추가: Tiny Box Loss, 13K Pre-training, 2단계 학습, P2 FPN)  
**다음 업데이트**: Phase 1 실행 후 결과 반영

**v2.0 업데이트 내역**:
- 🆕 Tiny Box Loss 가중치 부여 전략 추가 (Tier 1)
- 🆕 대규모 13K 외부 데이터 Pre-training 전략 추가 (Tier 2)
- 🆕 2단계 학습 파이프라인 (Curriculum Learning) 추가 (Tier 2)
- 🆕 P2 Feature Pyramid 레벨 추가 전략 추가 (Tier 2)
- 총 전략 수: 21개 → 25개로 확장
- 예상 개선폭: +0.6~1.2%p → +1.5~2.5%p로 상향

---

## 🚀 즉시 실행 명령어 모음

```bash
# === Phase 1: Quick Wins (5.25시간) 🔥 ===

# 1. Unclip Ratio 최적화 (10분)
cd /data/ephemeral/home/baseline_code
python scripts/optimize_unclip_ratio.py \
  --checkpoint checkpoints/kfold/fold_3/fold3_best.ckpt \
  --val_json kfold_results/fold_3/val.json \
  --ratio_range 1.85 2.15 --step 0.05

# 2. 🆕 Tiny Box Loss 가중치 구현 및 Fine-tuning (2시간)
# DBLossWeighted 클래스 구현 (ocr/models/loss/db_loss_weighted.py)
python runners/train.py \
    preset=hrnet_w44_1024_external_weighted_loss \
    model.loss.name=DBLossWeighted \
    model.loss.tiny_weight=10.0 \
    model.loss.small_weight=5.0 \
    ++datasets.train_dataset.annotation_path=train_augmented_full.json \
    ++datasets.val_dataset.annotation_path=kfold_results/fold_3/val.json \
    exp_name=hrnet_w44_tiny_box_weighted \
    trainer.max_epochs=10

# 3. 후처리 미세조정 (5분)
python scripts/postprocess_grid_search.py \
  --checkpoint checkpoints/hrnet_w44_tiny_box_weighted/best.ckpt \
  --thresh_range 0.210 0.220 --thresh_step 0.002 \
  --box_thresh_range 0.410 0.420 --box_thresh_step 0.005

# 4. WildReceipt + ICDAR 다운로드 및 변환 (3시간)
cd /data/ephemeral/home/data/pseudo_label
git clone https://github.com/clovaai/wildreceipt.git
wget https://rrc.cvc.uab.es/downloads/icdar2019_task1.zip && unzip icdar2019_task1.zip

cd /data/ephemeral/home/baseline_code
python scripts/convert_external_datasets.py \
  --wildreceipt ../data/pseudo_label/wildreceipt \
  --icdar ../data/pseudo_label/icdar2019_task1 \
  --output ../data/datasets/wildreceipt_icdar.json

python scripts/merge_datasets.py \
  --inputs train_augmented_full.json wildreceipt_icdar.json \
  --output train_extended_wildreceipt.json

# 5. 재훈련 (overnight)
python runners/train.py \
    preset=hrnet_w44_1024_extended \
    ++datasets.train_dataset.annotation_path=train_extended_wildreceipt.json \
    exp_name=hrnet_w44_extended_final \
    trainer.max_epochs=18

# === Phase 2: 고효과 실험 (25시간 추가) ⭐ ===

# 6. 🆕 대규모 13K Pre-training (8시간)
# 모든 외부 데이터셋 다운로드 및 통합
python scripts/convert_all_external_datasets.py \
  --sroie ../data/pseudo_label/sroie \
  --cord ../data/pseudo_label/cord-v2 \
  --wildreceipt ../data/pseudo_label/wildreceipt \
  --icdar ../data/pseudo_label/icdar2019_task1 \
  --output ../data/datasets/external_unified_13k.json

python runners/train.py \
    preset=hrnet_w44_1024_pretrain \
    ++datasets.train_dataset.annotation_path=external_unified_13k.json \
    exp_name=hrnet_w44_pretrain_13k \
    trainer.max_epochs=15

# 7. 🆕 2단계 학습 파이프라인 (6시간)
# Stage 1: 1024px Pre-training
python runners/train.py \
    preset=hrnet_w44_1024_stage1 \
    ++datasets.train_dataset.annotation_path=train_mega_dataset_13k.json \
    datasets.image_size=1024 \
    optimizer.lr=0.001 \
    trainer.max_epochs=15 \
    exp_name=stage1_pretrain_1024px_13k

# Stage 2: 1280px Fine-tuning
python runners/train.py \
    preset=hrnet_w44_1280_stage2 \
    ++resume_from=outputs/stage1_pretrain_1024px_13k/checkpoints/last.ckpt \
    ++datasets.train_dataset.annotation_path=train.json \
    datasets.image_size=1280 \
    optimizer.lr=0.0001 \
    trainer.max_epochs=8 \
    exp_name=stage2_finetune_1280px_competition

# 8. 🆕 P2 Feature Pyramid 추가 (5시간)
# FPNDecoderWithP2 구현 후
python runners/train.py \
    preset=hrnet_w44_1024_fpn_p2 \
    model.decoder.name=FPNDecoderWithP2 \
    ++datasets.train_dataset.annotation_path=train_augmented_full.json \
    dataloaders.train_dataloader.batch_size=2 \
    trainer.max_epochs=12 \
    exp_name=hrnet_w44_fpn_p2_small_objects

# 9. Multi-Scale TTA (30분)
python scripts/predict_multiscale_tta.py \
  --checkpoint checkpoints/stage2_finetune_1280px_competition/best.ckpt \
  --scales 960 1024 1088 \
  --weights 0.25 0.50 0.25

# 10. EfficientNet-B4 학습 (3시간)
python runners/train.py \
    preset=efficientnet_b4_1024 \
    ++datasets.train_dataset.annotation_path=train_extended_wildreceipt.json \
    exp_name=efficientnet_b4_extended

# 11. 혼합 앙상블 (1시간)
python scripts/mixed_backbone_ensemble.py \
  --model1 checkpoints/stage2_finetune_1280px_competition/best.ckpt \
  --model2 checkpoints/efficientnet_b4_extended/best.ckpt \
  --weights 0.55 0.45 --iou_thr 0.7
```

**예상 최종 성능: 99.7~100%+ H-Mean** 🎯🎯🎯

**🆕 신규 전략 요약**:
1. **Tiny Box Loss** - 작은 박스 검출 강화, +0.4~0.7%p
2. **13K Pre-training** - 대규모 데이터 일반화, +0.8~1.2%p
3. **2단계 학습** - Curriculum Learning, +0.3~0.6%p
4. **P2 FPN** - 고해상도 미세 텍스트, +0.2~0.5%p
