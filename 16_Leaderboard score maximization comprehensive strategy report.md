```markdown
# 리더보드 점수 극대화 종합 전략 보고서

**작성일**: 2026년 2월 2일  
**현재 최고 성능**: H-Mean **96.53%** (EfficientNet-B4 단일 모델)  
**최종 목표**: H-Mean **97.0%+** 달성  
**상태**: 🔥 핵심 돌파구 확보, 최종 단계 진입

---

## 📊 Executive Summary

### 핵심 성과

**단일 모델 최고 기록 달성 ⭐**
- **EfficientNet-B4**: 96.53% H-Mean (thresh=0.29)
- **ResNet50 5-Fold 앙상블**: 96.28% H-Mean
- **EfficientNet-B4가 ResNet50 앙상블을 단일 모델로 초과** (+0.25%p)

**주요 돌파구**
1. ✅ 백본 업그레이드 성공 (ResNet18 → ResNet50 → EfficientNet-B4)
2. ✅ Postprocessing 최적화 완료 (thresh=0.29 발견)
3. ✅ K-Fold 앙상블 구조 확립 (Voting≥3 검증)
4. 🔄 WandB Sweep 진행 중 (Learning Rate 최적화)

**다음 목표**
- 단기: EfficientNet-B4 5-Fold 앙상블로 **96.70%** 달성
- 최종: 혼합 앙상블 및 TTA로 **97.0%+** 돌파

---

## 🔍 현재 상황 분석

### 1. 성능 진화 타임라인

| 단계 | 모델 | H-Mean | Precision | Recall | 핵심 변화 |
|------|------|--------|-----------|--------|----------|
| Baseline | ResNet18 (960px) | 95.81% | - | - | 초기 베이스라인 |
| Phase 1 | ResNet50 Basic | 96.20% | 96.53% | 95.87% | 백본 업그레이드 (+0.39%p) |
| Phase 2 | ResNet50 Aggressive | 96.26% | 96.49% | 96.23% | 후처리 최적화 (+0.06%p) |
| Phase 3 | ResNet50 5-Fold | 96.28% | 96.45% | 96.29% | 앙상블 안정성 (+0.02%p) |
| **Phase 4** | **EfficientNet-B4** | **96.53%** | **96.94%** | **96.36%** | **백본 혁신 (+0.25%p)** ⭐ |

**총 개선폭**: 95.81% → 96.53% = **+0.72%p** (상대 개선 +0.75%)

### 2. 병목 지점 진단

**✅ 해결된 문제**
- ~~Precision-Recall 불균형~~ → 0.58%p gap (양호)
- ~~ResNet50 성능 천장~~ → EfficientNet-B4로 돌파
- ~~Postprocessing 파라미터~~ → thresh=0.29 최적값 발견

**🎯 현재 과제**
1. **Recall 추가 개선 필요**: 96.36% → 96.5%+ 목표
2. **앙상블 효과 극대화**: 단일 96.53% → 5-Fold 96.70%+ 목표
3. **최종 0.5%p 갭 극복**: 96.5% → 97.0% 돌파 전략

### 3. 모델별 성능 비교

| 모델 | H-Mean | Precision | Recall | P-R Gap | 박스 수 | 특징 |
|------|--------|-----------|--------|---------|---------|------|
| **EfficientNet-B4** | **96.53%** | **96.94%** | 96.36% | 0.58%p | ~45,200 | 🥇 최고 성능 |
| ResNet50 5-Fold | 96.28% | 96.45% | 96.29% | 0.16%p | 45,010 | 🥈 안정적 |
| ResNet50 Aggressive | 96.26% | 96.49% | 96.23% | 0.26%p | 44,628 | 🥉 단일 최고 |
| ResNet50 Basic | 96.20% | 96.53% | 95.87% | 0.66%p | - | 기준선 |

**핵심 인사이트**
- EfficientNet-B4의 Precision이 가장 높음 (96.94%)
- ResNet50 5-Fold의 P-R Gap이 가장 낮음 (0.16%p)
- **EfficientNet-B4 5-Fold 조합이 최적**일 가능성 높음

---

## 🎯 리더보드 극대화 전략

### 전략 A: EfficientNet-B4 5-Fold 앙상블 (최우선 ⭐⭐⭐)

**목표**: H-Mean **96.70%** 달성

**실행 계획**

**Step 1: WandB Sweep 완료 및 최적 설정 확정** (진행 중)
```

# 현재 Sweep ID: v5inrfwe

# 탐색 중인 파라미터:

- Learning Rate: 0.0003 ~ 0.0007
- Weight Decay: 0.0001 ~ 0.0002
- Fixed: thresh=0.29, box_thresh=0.25

# 예상 최적 설정:

optimal_lr: 0.00045

optimal_wd: 0.00015

```

**예상 소요**: 24시간 (Sweep 완료)

**Step 2: 최적 설정으로 5-Fold 학습**
```

# 각 Fold 순차 학습

for fold in {0..4}; do

python runners/[train.py](http://train.py) \

preset=efficientnet_b4_optimal \

model.encoder.model_name=tf_efficientnet_b4 \

datasets.image_size=1024 \

[optimizer.lr](http://optimizer.lr)=0.00045 \

optimizer.weight_decay=0.00015 \

++datasets.train_dataset.annotation_path=kfold_results_v2/fold_${fold}/train.json \

++datasets.val_dataset.annotation_path=kfold_results_v2/fold_${fold}/val.json \

exp_name=efficientnet_b4_optimal_fold${fold}

done

```

**예상 소요**: 25시간 (Fold당 5시간 × 5)

**Step 3: Voting≥3 앙상블**
```

python scripts/ensemble_[5fold.py](http://5fold.py) \

--fold0 outputs/efficientnet_b4_optimal_fold0/submissions/*.json \

--fold1 outputs/efficientnet_b4_optimal_fold1/submissions/*.json \

--fold2 outputs/efficientnet_b4_optimal_fold2/submissions/*.json \

--fold3 outputs/efficientnet_b4_optimal_fold3/submissions/*.json \

--fold4 outputs/efficientnet_b4_optimal_fold4/submissions/*.json \

--output outputs/ensemble_effb4_5fold_voting3.json \

--voting 3 \

--iou 0.5

```

**예상 성능**
```

H-Mean:    96.65 ~ 96.75%

Precision: 96.85 ~ 96.95%

Recall:    96.45 ~ 96.60%

```

**성공 확률**: 90% (단일 모델이 이미 96.53%이므로 앙상블 효과 확실)

---

### 전략 B: ResNet50 + EfficientNet-B4 혼합 앙상블 (차선책 ⭐⭐)

**목표**: H-Mean **96.75%** 달성 (아키텍처 다양성 활용)

**근거**
- ResNet50: 안정적, Large text 강함
- EfficientNet-B4: Small text 강함, 고해상도 최적화
- **상호 보완 효과 기대**

**실행 계획**

**10-Fold 혼합 앙상블**
```

# 10개 모델 조합

models = [

# ResNet50 (기존 5 folds, H-Mean 96.28%)

'resnet50_fold0_aggressive.json',

'resnet50_fold1_aggressive.json',

'resnet50_fold2_aggressive.json',

'resnet50_fold3_aggressive.json',

'resnet50_fold4_aggressive.json',

# EfficientNet-B4 (신규 5 folds, 예상 96.55~96.60%)

'efficientnet_b4_fold0.json',

'efficientnet_b4_fold1.json',

'efficientnet_b4_fold2.json',

'efficientnet_b4_fold3.json',

'efficientnet_b4_fold4.json',

]

# Voting≥6 (60% 합의)

python scripts/ensemble_mixed_[backbone.py](http://backbone.py) \

--models ${models[@]} \

--voting 6 \

--iou 0.5 \

--output ensemble_mixed_10fold_voting6.json

```

**가중 앙상블 (고급 버전)**
```

# 백본별 가중치 차등

resnet50_weight = 0.45  # 5개 모델

efficientnet_b4_weight = 0.55  # 5개 모델 (더 높은 성능)

# Weighted Box Fusion

from ensemble_boxes import weighted_boxes_fusion

final_boxes = weighted_boxes_fusion(

boxes_list,

scores_list,

weights=[resnet50_weight]*5 + [efficientnet_b4_weight]*5,

iou_thr=0.5,

skip_box_thr=0.3

)

```

**예상 성능**
```

H-Mean:    96.70 ~ 96.80%

Precision: 96.80 ~ 96.90%

Recall:    96.60 ~ 96.75%

```

**장점**
- 백본 다양성으로 더 robust한 예측
- Small text와 Large text 모두 커버
- 각 백본의 강점 활용

**예상 소요**: 1시간 (앙상블만, EfficientNet-B4 학습 완료 후)

---

### 전략 C: Test-Time Augmentation (TTA) (보조 전략 ⭐)

**목표**: Recall **+0.3~0.5%p** 추가 개선

**TTA 파이프라인**
```

tta_transforms = [

'original',           # 원본

'horizontal_flip',    # 좌우 반전

'scale_1.1',         # 10% 확대 (작은 글씨)

'brightness_up',     # 밝게 (어두운 영수증)

'sharpen',           # 선명하게 (흐릿한 글씨)

]

# 각 이미지당 5개 예측 생성 → WBF로 융합

for image in test_images:

predictions = []

for transform in tta_transforms:

augmented = apply_transform(image, transform)

pred = model(augmented)

pred_original_space = reverse_transform(pred, transform)

predictions.append(pred_original_space)

# Weighted Box Fusion

final = weighted_boxes_fusion(predictions, iou_thr=0.5)

```

**적용 대상**
- EfficientNet-B4 5-Fold 앙상블 결과에 추가 적용
- 또는 최고 성능 단일 모델에 적용

**예상 효과**
```

Before TTA: 96.70% (5-Fold 앙상블)

After TTA:  96.90 ~ 97.00%

Recall 개선: +0.3 ~ 0.5%p

```

**주의사항**
- HorizontalFlip은 박스 좌표 x축 반전 필수
- Scale 변환 시 박스 좌표 스케일 조정 필요
- 이전 TTA 실패 사례를 참고하여 구현

**예상 소요**: 2시간 (구현 1시간 + 실행 1시간)

---

### 전략 D: Soft Voting & 구제 메커니즘 (미세 조정)

**목표**: Voting≥3의 Recall 손실 보완

**현재 문제**
- Voting≥3: 엄격한 기준 → 일부 True Positive 누락
- 2/5 투표 박스 중 고신뢰도 박스는 구제 필요

**Soft Voting 알고리즘**
```

def soft_voting_with_rescue(predictions, min_votes=3, rescue_threshold=0.95):

"""

(Votes >= 3) OR (Votes == 2 AND Mean_Score > 0.95)

"""

for box_group in grouped_boxes:

vote_count = len(box_group)

avg_score = np.mean([box['score'] for box in box_group])

# 기본 조건: 3표 이상

if vote_count >= min_votes:

final_boxes.append(average_box(box_group))

# 구제 조건: 2표 + 고신뢰도

elif vote_count == 2 and avg_score > rescue_threshold:

final_boxes.append(average_box(box_group))

return final_boxes

```

**예상 효과**
```

기존 Voting≥3: Recall 96.29% (ResNet50)

Soft Voting:   Recall 96.45 ~ 96.55% (+0.15~0.25%p)

H-Mean:        96.35 ~ 96.40%

```

**적용 시기**: EfficientNet-B4 5-Fold 앙상블에 적용

---

## 📈 단계별 실행 로드맵

### Phase 1: EfficientNet-B4 5-Fold 학습 (2일)

**Day 1-2: 학습 파이프라인 실행**
- [🔄] WandB Sweep 완료 대기 (24시간)
- [ ] 최적 하이퍼파라미터 확정
- [ ] 5-Fold 학습 스크립트 실행
  - Fold 0~4 순차 학습 (또는 병렬)
  - 각 Fold: 5시간 × 5 = 25시간
  - 체크포인트 자동 저장

**성공 기준**: 각 Fold가 Val H-Mean 96.55%+ 달성

---

### Phase 2: 기본 앙상블 전략 (1일)

**Day 3: Voting≥3 앙상블**
- [ ] 5개 Fold 예측 생성 (각 5분 × 5 = 25분)
- [ ] Voting≥3 앙상블 실행
- [ ] CSV 변환 및 리더보드 제출

**목표 성능**: H-Mean **96.65 ~ 96.75%**

**Day 3 오후: Soft Voting 실험**
- [ ] Soft Voting 스크립트 구현
- [ ] rescue_threshold 파라미터 탐색 (0.90, 0.93, 0.95)
- [ ] 최고 성능 설정으로 제출

**목표 성능**: H-Mean **96.70 ~ 96.80%**

---

### Phase 3: 혼합 앙상블 전략 (1일)

**Day 4: 백본 혼합 앙상블**
- [ ] ResNet50 5-Fold + EfficientNet-B4 5-Fold 결합
- [ ] Voting≥6 (10개 중 6개 합의) 실험
- [ ] Weighted 앙상블 (백본별 가중치) 실험
- [ ] 최고 성능 제출

**목표 성능**: H-Mean **96.75 ~ 96.85%**

---

### Phase 4: TTA 및 최종 최적화 (1일)

**Day 5: Test-Time Augmentation**
- [ ] TTA 파이프라인 구현 (좌표 변환 검증)
- [ ] 5-TTA × 최고 앙상블 조합
- [ ] 최종 제출

**목표 성능**: H-Mean **96.90 ~ 97.00%** 🎯

**Day 5 오후: 파라미터 미세 조정**
- [ ] Postprocessing 미세 조정 (thresh=0.285 ~ 0.295)
- [ ] IoU threshold 조정 (0.45 ~ 0.55)
- [ ] 최종 제출

**최종 목표**: H-Mean **97.0%+** 달성

---

## 🎲 예상 성능 시나리오

### 낙관적 시나리오 (70% 확률)

| 단계 | 전략 | H-Mean | 누적 개선 |
|------|------|--------|----------|
| 현재 | EfficientNet-B4 단일 | 96.53% | - |
| Phase 1 | 5-Fold Voting≥3 | 96.70% | +0.17%p |
| Phase 2 | Soft Voting | 96.78% | +0.25%p |
| Phase 3 | 혼합 앙상블 | 96.88% | +0.35%p |
| Phase 4 | TTA | **97.05%** | **+0.52%p** ✅ |

**최종**: H-Mean **97.0%+** 달성 🎯

---

### 현실적 시나리오 (90% 확률)

| 단계 | 전략 | H-Mean | 누적 개선 |
|------|------|--------|----------|
| 현재 | EfficientNet-B4 단일 | 96.53% | - |
| Phase 1 | 5-Fold Voting≥3 | 96.68% | +0.15%p |
| Phase 2 | Soft Voting | 96.73% | +0.20%p |
| Phase 3 | 혼합 앙상블 | 96.82% | +0.29%p |
| Phase 4 | TTA | **96.95%** | **+0.42%p** |

**최종**: H-Mean **96.9%+** 달성 (목표에 근접)

---

### 보수적 시나리오 (95% 확률)

| 단계 | 전략 | H-Mean | 누적 개선 |
|------|------|--------|----------|
| 현재 | EfficientNet-B4 단일 | 96.53% | - |
| Phase 1 | 5-Fold Voting≥3 | 96.65% | +0.12%p |
| Phase 2 | Soft Voting | 96.70% | +0.17%p |
| Phase 3 | 혼합 앙상블 | 96.78% | +0.25%p |
| Phase 4 | TTA | **96.88%** | **+0.35%p** |

**최종**: H-Mean **96.85%+** 달성 (안정적 개선)

---

## ⚠️ 리스크 관리

### 주요 리스크 및 대응 방안

**Risk 1: WandB Sweep이 현재 설정보다 나쁜 결과**
- **확률**: 20%
- **영향**: 5-Fold 성능이 단일 모델(96.53%)에 못 미침
- **대응**: 
  - 현재 설정(thresh=0.29, box_thresh=0.25)을 기본값으로 사용
  - Sweep에서 상위 3개 설정으로 각각 1개 Fold 학습 후 검증
  - 최고 설정으로 전체 5-Fold 재학습

**Risk 2: 5-Fold 앙상블 효과가 미미함**
- **확률**: 30%
- **영향**: 단일 96.53% → 앙상블 96.60% (기대 이하)
- **대응**: 
  - 즉시 혼합 앙상블(전략 B)로 전환
  - TTA를 단일 모델에 적용하여 보완
  - Soft Voting으로 Recall 복구

**Risk 3: TTA 구현 오류 (좌표 변환)**
- **확률**: 15%
- **영향**: 성능 급락 (이전 TTA 실패 사례 재발)
- **대응**: 
  - 철저한 좌표 변환 검증 (시각화)
  - HorizontalFlip부터 단계적 검증
  - 실패 시 TTA 없이 앙상블만으로 진행

**Risk 4: Public-Private 리더보드 차이**
- **확률**: 40%
- **영향**: Public에서 좋아도 Private에서 하락
- **대응**: 
  - K-Fold 교차 검증 점수를 신뢰
  - 과도한 Public 최적화 지양
  - 일반화 성능 중시 (TTA, 앙상블)

---

## 🎯 핵심 성공 요인

### Critical Success Factors

**1. EfficientNet-B4 5-Fold 품질 (최우선)**
- 각 Fold가 단일 모델 수준(96.5%+) 달성 필수
- Fold 간 다양성 확보 (K-Fold 데이터 분할)
- 과적합 방지 (Early Stopping, Validation 모니터링)

**2. 앙상블 로직 정교함**
- IoU threshold 최적화 (0.5 고정 vs 탐색)
- Voting threshold 실험 (≥3, ≥4, Soft Voting)
- Weighted Box Fusion 활용

**3. Precision-Recall 균형 유지**
- 현재 EfficientNet-B4는 0.58%p gap (양호)
- 앙상블 후에도 gap < 0.5%p 유지 목표
- Recall 복구 시 Precision 하락 최소화

**4. 실험 속도와 효율성**
- GPU 자원 효율적 활용 (Mixed Precision, Gradient Accumulation)
- 병렬 학습 가능하면 활용 (5-Fold 동시 학습)
- 빠른 검증 사이클 (각 전략을 즉시 검증)

---

## 📋 실행 체크리스트

### Week 1: EfficientNet-B4 5-Fold (최우선)

**Day 1**
- [🔄] WandB Sweep 완료 확인 (현재 진행 중)
- [ ] Sweep 결과 분석 및 최적 설정 확정
- [ ] 최적 설정으로 Fold 0 재학습 시작

**Day 2**
- [ ] Fold 0 완료 확인 (Val H-Mean 96.55%+ 확인)
- [ ] Fold 1, 2 학습 시작 (병렬 가능하면 동시)
- [ ] 학습 진행 모니터링 (WandB 대시보드)

**Day 3**
- [ ] Fold 3, 4 학습 시작
- [ ] 완료된 Fold들로 예측 생성
- [ ] 부분 앙상블 테스트 (3-Fold Voting≥2)

**Day 4**
- [ ] 전체 5-Fold 학습 완료 확인
- [ ] 5개 Fold 예측 생성
- [ ] Voting≥3 앙상블 실행 및 제출
- [ ] **목표: H-Mean 96.70%** 🎯

**Day 5**
- [ ] Soft Voting 구현 및 실험
- [ ] rescue_threshold 최적화
- [ ] 최고 설정으로 제출
- [ ] **목표: H-Mean 96.75%** 🎯

---

### Week 2: 혼합 앙상블 & TTA (최종 단계)

**Day 6**
- [ ] ResNet50 + EfficientNet-B4 10-Fold 앙상블
- [ ] Voting≥6 실험
- [ ] Weighted 앙상블 실험
- [ ] **목표: H-Mean 96.80%** 🎯

**Day 7**
- [ ] TTA 파이프라인 구현
- [ ] 좌표 변환 검증 (시각화)
- [ ] HorizontalFlip 단독 테스트

**Day 8**
- [ ] 5-TTA 전체 적용
- [ ] 최고 앙상블에 TTA 결합
- [ ] 최종 제출
- [ ] **목표: H-Mean 96.90 ~ 97.00%** 🎯

**Day 9-10 (버퍼)**
- [ ] 파라미터 미세 조정
- [ ] 추가 실험 (필요 시)
- [ ] 최종 검증 및 문서화

---

## 💡 추가 개선 아이디어 (선택적)

### 고급 전략 (시간 여유 시)

**1. Multi-Resolution 앙상블**
```

# 다양한 해상도 모델 조합

models = [

'efficientnet_b4_896px',   # 빠른 추론

'efficientnet_b4_1024px',  # 현재 사용

'efficientnet_b4_1280px',  # 작은 글씨 특화

]

# 해상도별 강점 활용

# 896px: 일반 텍스트

# 1024px: 균형

# 1280px: 작은 텍스트

```

**예상 효과**: +0.1 ~ 0.2%p

**2. Pseudo-Labeling**
```

# 외부 영수증 데이터로 Pre-training

# → 본 데이터로 Fine-tuning

# 주의: Public 과적합 위험

```

**예상 효과**: +0.2 ~ 0.3%p (데이터 품질에 따라)

**3. 박스 후처리 정교화**
```

# 이상치 박스 제거

def filter_outlier_boxes(boxes):

# 너무 작은 박스 제거 (< 50 pixels)

# 너무 긴 박스 제거 (aspect ratio > 20)

# 겹침이 과도한 박스 제거 (IoU > 0.95)

pass

```

**예상 효과**: +0.05 ~ 0.1%p

---

## 📊 예상 리더보드 진입

### 최종 목표 달성 시 예상 순위

**가정**: 현재 리더보드 분포 기준

| H-Mean | 예상 순위 | 달성 전략 |
|--------|----------|---------|
| 96.53% | Top 30% | 현재 위치 (EfficientNet-B4 단일) |
| 96.70% | Top 20% | 5-Fold 앙상블 |
| 96.85% | Top 10% | 혼합 앙상블 |
| **97.00%** | **Top 5%** | **TTA + 최적화** 🎯 |
| 97.20%+ | Top 1% | (추가 혁신 필요) |

**목표**: **Top 5% 진입** (H-Mean 97.0%)

---

## 🎓 핵심 교훈 정리

### 검증된 성공 패턴

**1. 백본 업그레이드가 가장 효과적**
- ResNet18 → ResNet50: +0.39%p
- ResNet50 → EfficientNet-B4: +0.25%p
- **총 +0.64%p** (전체 개선의 89%)

**2. Postprocessing 최적화는 빠르고 확실**
- 재학습 불필요
- thresh=0.29 발견으로 +0.16%p
- 각 모델마다 최적값 다름 (탐색 필수)

**3. 앙상블은 안정성 확보**
- 단일 모델 96.26% → 5-Fold 96.28% (미미)
- 하지만 Precision-Recall 균형 개선 (gap 0.26%p → 0.16%p)
- **EfficientNet-B4 5-Fold는 더 큰 효과 기대**

**4. Precision-Recall 균형이 핵심**
- Gap < 0.5%p 유지 시 H-Mean 최대화
- Voting≥3 (과반수)가 최적 균형점
- Soft Voting으로 미세 조정 가능

### 실패 패턴 및 교훈

**1. TTA 구현 오류는 치명적**
- HorizontalFlip 좌표 변환 누락 → -18%p
- **교훈**: 좌표 변환을 철저히 검증, 시각화 필수

**2. 과도한 Postprocessing은 역효과**
- Ultra-aggressive (thresh=0.20) → -0.17%p
- **교훈**: 적절한 균형점 찾기, Validation 기반 탐색

**3. 낮은 Voting threshold는 노이즈 증가**
- Voting≥1 → Precision 94.53% (-1.96%p)
- **교훈**: 과반수 투표(≥3) 원칙 고수

---

## 🚀 최종 결론

### 핵심 메시지

**현재 상황**: EfficientNet-B4로 **96.53%** 달성, 핵심 돌파구 확보 ✅

**최우선 과제**: EfficientNet-B4 5-Fold 앙상블 → **96.70%** 목표 🎯

**최종 목표**: 혼합 앙상블 + TTA → **97.0%** 달성 🏆

### Next Actions (우선순위)

**🔥 즉시 실행**
1. WandB Sweep 완료 확인 (24시간 내)
2. 최적 설정으로 Fold 0 재학습 시작

**📈 Week 1 목표**
3. 5-Fold 전체 학습 완료
4. Voting≥3 앙상블 제출 → **96.70%** 달성
5. Soft Voting 실험 → **96.75%** 도전

**🎯 Week 2 목표**
6. 혼합 앙상블 (ResNet50 + EfficientNet-B4) → **96.80%** 달성
7. TTA 구현 및 적용 → **96.90 ~ 97.00%** 최종 목표

### 성공 확률 평가

- **96.70% 달성**: 90% 확률 (5-Fold 앙상블)
- **96.85% 달성**: 70% 확률 (혼합 앙상블)
- **97.00% 달성**: 50% 확률 (TTA + 최적화)

**전략**: 단계별로 안정적 개선, 각 단계마다 제출하여 리스크 분산

---

## 📁 산출물 목록

### 제출 예정 파일

```

submissions/

├── submission_effb4_5fold_voting3.csv          # Phase 1 (목표: 96.70%)

├── submission_effb4_5fold_soft_voting.csv      # Phase 2 (목표: 96.75%)

├── submission_mixed_10fold_voting6.csv         # Phase 3 (목표: 96.80%)

├── submission_mixed_10fold_weighted.csv        # Phase 3 (대안)

├── submission_final_with_tta.csv               # Phase 4 (목표: 97.00%)

└── submission_final_optimized.csv              # Phase 4 (최종)

```

### 실험 기록

```

outputs/

├── efficientnet_b4_optimal_fold{0-4}/         # 5-Fold 체크포인트

├── ensemble_effb4_5fold_voting3.json          # 앙상블 중간 결과

├── ensemble_mixed_10fold.json                  # 혼합 앙상블

└── predictions_with_tta/                       # TTA 예측 결과

```

---

**보고서 작성**: 2026년 2월 2일  
**다음 업데이트**: 5-Fold 학습 완료 후  
**최종 목표**: H-Mean **97.0%** 🎯  
**상태**: ✅ 전략 수립 완료, 실행 준비 완료
```