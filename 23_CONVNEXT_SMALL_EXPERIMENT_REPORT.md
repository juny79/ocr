# ConvNeXt-Small Experiment Report
## 예상과 다른 결과: 모델 크기와 성능의 역설

**작성일**: 2026년 2월 3일  
**실험자**: OCR Team  
**목적**: ConvNeXt-Small (50M) 최적화 파라미터 검증 및 성능 분석

---

## 📊 Executive Summary

### 핵심 발견
ConvNeXt-Small이 **예상과 달리 가장 낮은 리더보드 성능**을 기록했습니다. 더 큰 모델이 더 작은 모델들(Tiny, EfficientNet-B3)보다 낮은 성능을 보이는 **역설적 결과**가 나타났습니다.

### 최종 리더보드 결과 (ConvNeXt-Small Epoch 18)
```
H-Mean:    95.98% (3위 / 최하위)
Precision: 97.39% (매우 높음 ⚠️)
Recall:    94.83% (매우 낮음 ⚠️)
```

### 전체 모델 비교
| 순위 | Model | Params | Val H-Mean | LB H-Mean | Gap | 특징 |
|------|-------|--------|------------|-----------|-----|------|
| 🥇 1위 | **ConvNeXt-Tiny** | 28M | 96.18% | **96.25%** | **+0.07%p** | 언더피팅 우위 |
| 🥈 2위 | EfficientNet-B3 | 12M | 96.58% | 96.19% | -0.39%p | 오버피팅 |
| 🥉 3위 | **ConvNeXt-Small** | 50M | 96.13% | **95.98%** | **-0.15%p** | 과도한 규제? |

**결론**: 50M 파라미터 모델이 28M, 12M 모델보다 낮은 성능 → **Capacity와 성능이 비례하지 않음**

---

## 🎯 실험 설계

### 1. 모델 아키텍처
```yaml
Model: ConvNeXt-Small
Parameters: ~50M (4x EfficientNet-B3, 1.8x ConvNeXt-Tiny)
Encoder: convnext_small
  - Stages: [3, 3, 27, 3]  # Tiny는 [3, 3, 9, 3]
  - Channels: [96, 192, 384, 768]
  - Depth: 36 total blocks (vs Tiny's 18)
```

### 2. 최적화 파라미터 (ConvNeXt-Tiny 대비)
```python
Learning Rate:    0.0004    # -11% (0.00045 → 0.0004, baseline으로 회귀)
Weight Decay:     0.00012   # +41% (0.000085 → 0.00012, 강한 규제)
T_max:            20
eta_min:          0.000008
Max Epochs:       20
Early Stopping:   patience=5
```

**전략 근거**: 
- 더 큰 모델 → 더 강한 규제 필요
- Baseline LR로 안정성 확보
- ConvNeXt-Tiny의 언더피팅 장점 유지

### 3. Progressive Resolution
```
Epoch 0-3:  640px × 640px
Epoch 4+:   960px × 960px (해상도 증가)
```

---

## 📈 학습 진행 과정

### Phase 1: 저해상도 학습 (Epoch 0-3)
```
Epoch 0: Val H-Mean = 낮은 초기 성능
Epoch 1-3: 점진적 개선
```

### Phase 2: 고해상도 전환 (Epoch 4)
```
🔄 Resolution Switch: 640px → 960px
- 초기 성능 하락 (예상된 현상)
- 이후 회복 및 개선
```

### Phase 3: 수렴 단계 (Epoch 5-19)
```
Epoch 15: Val H-Mean = 84.96% (중간 체크포인트)
Epoch 18: Val H-Mean = 96.13% ⭐ BEST
Epoch 19: Val H-Mean = 96.00%
Epoch 20: 학습 완료
```

### 최종 Test 성능 (Fold 0)
```
Test H-Mean:    96.13%
Test Precision: 95.98%
Test Recall:    96.47%
```

**관찰**: Test에서는 높은 Recall (96.47%), 하지만 리더보드에서는 낮은 Recall (94.83%)

---

## 🔍 결과 분석

### 1. 예상 vs 실제 성능

#### 가설 (실험 전)
```
ConvNeXt-Small (50M)
  > ConvNeXt-Tiny (28M)  
  > EfficientNet-B3 (12M)

근거:
- 더 많은 파라미터 → 더 높은 capacity
- 더 깊은 네트워크 → 더 나은 feature extraction
- 최적화된 regularization → 오버피팅 방지
```

#### 실제 결과
```
ConvNeXt-Tiny (28M): 96.25% 🥇
  > EfficientNet-B3 (12M): 96.19% 🥈
  > ConvNeXt-Small (50M): 95.98% 🥉

실제:
- 더 큰 모델이 더 낮은 성능
- 가장 작은 모델(Tiny)이 최고 성능
- Weight decay 증가가 역효과?
```

### 2. Validation ↔ Leaderboard Gap 분석

| Model | Val | LB | Gap | 해석 |
|-------|-----|----|----|------|
| **ConvNeXt-Tiny** | 96.18% | 96.25% | **+0.07%p** | ✅ 언더피팅, 좋은 일반화 |
| EfficientNet-B3 | 96.58% | 96.19% | -0.39%p | ⚠️ 오버피팅 |
| **ConvNeXt-Small** | 96.13% | 95.98% | **-0.15%p** | ⚠️ 약한 오버피팅 |

**인사이트**:
- ConvNeXt-Small도 음의 갭(-0.15%p) → 오버피팅 징후
- Weight decay 0.00012가 충분하지 않았을 가능성
- ConvNeXt-Tiny의 0.000085가 오히려 최적값이었을 수 있음

### 3. Precision vs Recall 불균형

```
ConvNeXt-Small Leaderboard:
  Precision: 97.39% ← 매우 높음 (보수적)
  Recall:    94.83% ← 매우 낮음 (많은 놓침)
  
비교:
  ConvNeXt-Tiny:
    Precision: 96.67%
    Recall:    95.99%
    → 균형잡힌 성능

  EfficientNet-B3:
    Precision: 96.33%
    Recall:    96.06%
    → 균형잡힌 성능
```

**문제 진단**:
- **Precision 과다, Recall 부족** → 모델이 너무 보수적으로 예측
- False Negative 증가 (실제 텍스트를 놓침)
- Weight decay 0.00012가 **과도한 규제**로 작용했을 가능성

### 4. 상세 성능 비교표

| Metric | ConvNeXt-Small | ConvNeXt-Tiny | Diff | 분석 |
|--------|----------------|---------------|------|------|
| **LB H-Mean** | 95.98% | 96.25% | **-0.27%p** | Tiny가 명확히 우수 |
| **LB Precision** | 97.39% | 96.67% | +0.72%p | Small이 과도하게 보수적 |
| **LB Recall** | 94.83% | 95.99% | **-1.16%p** | Small이 많이 놓침 ⚠️ |
| **Val H-Mean** | 96.13% | 96.18% | -0.05%p | 거의 동일 |
| **Parameters** | 50M | 28M | +78% | 1.8배 더 큼 |
| **Val→LB Gap** | -0.15%p | +0.07%p | -0.22%p | Tiny가 일반화 우수 |

---

## 🧪 실패 원인 분석

### 1. 과도한 Regularization (가장 유력)

#### Weight Decay 비교
```python
ConvNeXt-Tiny:  0.000085  → LB 96.25% ✅
ConvNeXt-Small: 0.00012   → LB 95.98% ❌
  (+41% 증가)
```

**분석**:
- Weight decay를 41% 증가시켰으나 역효과
- 50M 파라미터 모델에는 **0.00012가 너무 강한 규제**
- 모델이 과도하게 보수적으로 학습 → Recall 하락 (94.83%)
- 적절한 값은 0.00009 ~ 0.0001 정도였을 것으로 추정

#### Precision/Recall 불균형의 근본 원인
```
높은 Weight Decay (0.00012)
  → 가중치를 과도하게 억제
  → 모델이 확신있는 경우만 예측
  → Precision 상승 (97.39%), Recall 하락 (94.83%)
  → H-Mean 하락 (95.98%)
```

### 2. Model Capacity vs Dataset Size 불일치

```
Dataset Size: 2,618 images (train)
Model Capacity:
  - EfficientNet-B3: 12M params  → LB 96.19%
  - ConvNeXt-Tiny:   28M params  → LB 96.25% ✅
  - ConvNeXt-Small:  50M params  → LB 95.98% ❌

결론: 2.6K 이미지에는 28M이 최적, 50M은 과도
```

**Sweet Spot**: 28M parameters (ConvNeXt-Tiny)
- 충분한 capacity + 적절한 regularization
- 데이터셋 크기와 균형

### 3. Progressive Resolution의 역효과 가능성

```
640px (Epoch 0-3) → 960px (Epoch 4+)

문제점:
- 해상도 전환 후 고해상도 학습이 18 epoch만 진행
- 960px에서의 충분한 수렴 시간 부족?
- Tiny는 전체 20 epoch 동안 동일 전략 사용
```

**가설**: 
- Small은 더 큰 모델이라 960px 해상도에서 더 긴 학습 필요
- Epoch 18에서 멈춘 것이 최적이 아니었을 가능성

### 4. Learning Rate 전략

```python
ConvNeXt-Tiny:  lr=0.00045  # 공격적
ConvNeXt-Small: lr=0.0004   # 보수적 (baseline)

의도: 안정적 학습
실제: 학습 속도 저하 + weight decay와 결합되어 과도한 억제
```

---

## 💡 핵심 인사이트

### 1. "Bigger is NOT Always Better"
```
❌ 잘못된 가정: 더 큰 모델 = 더 좋은 성능
✅ 실제: Dataset size와 Model capacity의 균형이 중요

최적 모델 크기:
  12M (B3): 약간 작음 → 96.19%
  28M (Tiny): 최적 ⭐ → 96.25%
  50M (Small): 과도 → 95.98%
```

### 2. Regularization Paradox
```
Weight Decay 증가 전략이 실패:
  더 큰 모델 → 더 강한 규제 (전통적 접근)
  실제: 더 강한 규제 → 성능 하락

이유:
  - 데이터셋이 작음 (2.6K images)
  - 강한 규제가 모델의 표현력을 과도하게 제한
  - Recall 희생 (94.83%), Precision 과잉 (97.39%)
```

### 3. Validation ≠ Leaderboard
```
Validation에서 유사한 성능:
  ConvNeXt-Tiny:  96.18%
  ConvNeXt-Small: 96.13%
  Diff: -0.05%p (거의 동일)

Leaderboard에서 큰 차이:
  ConvNeXt-Tiny:  96.25%
  ConvNeXt-Small: 95.98%
  Diff: -0.27%p (명확한 차이)

교훈: 
  Validation 성능만으로는 실제 일반화 성능을 알 수 없음
  Tiny의 +0.07%p 갭이 Small의 -0.15%p 갭보다 중요한 지표
```

### 4. 언더피팅이 오버피팅보다 나을 수 있다
```
ConvNeXt-Tiny: 
  - Validation 96.18% (상대적으로 낮음)
  - Leaderboard 96.25% (최고)
  - 언더피팅 → 테스트 셋에 더 잘 일반화

ConvNeXt-Small:
  - Validation 96.13% (Tiny와 유사)
  - Leaderboard 95.98% (최하)
  - 약한 오버피팅 → 일반화 능력 부족
```

---

## 📋 실험 타임라인 전체 복기

### Timeline: EfficientNet-B3 → ConvNeXt-Tiny → ConvNeXt-Small

#### Phase 1: EfficientNet-B3 (Baseline)
```
Training: 20 epochs
Best: Epoch 11, Val 96.58%
Leaderboard: 96.19% (-0.39%p)
문제점: 명확한 오버피팅
```

#### Phase 2: ConvNeXt-Tiny (Architecture Change)
```
Training: 20 epochs
Best: Epoch 7, Val 96.18%
Leaderboard: 96.25% (+0.07%p) ⭐
성공요인: 
  - 언더피팅 우위
  - Early stopping 효과
  - 적절한 capacity (28M)
```

#### Phase 3: ConvNeXt-Small (Scale Up + Optimization)
```
Training: 20 epochs
Best: Epoch 18, Val 96.13%
Leaderboard: 95.98% (-0.15%p) ❌
실패요인:
  - 과도한 regularization
  - Capacity 과잉 (50M)
  - Precision/Recall 불균형
```

---

## 🎓 Lessons Learned

### 1. 데이터셋 크기에 맞는 모델 선택
```
✅ DO:
  - 작은 데이터셋(~3K) → 중간 크기 모델(28M)
  - Validation gap 패턴 관찰 (+ vs -)
  - Precision/Recall 균형 체크

❌ DON'T:
  - 무조건 큰 모델 선택
  - Validation 성능만 보고 판단
  - 과도한 regularization 적용
```

### 2. Regularization은 양날의 검
```
적절한 Weight Decay:
  - 12M (B3):   0.000085 → 오버피팅
  - 28M (Tiny): 0.000085 → 최적 ⭐
  - 50M (Small): 0.00012 → 과도한 규제

교훈:
  "더 큰 모델 = 더 강한 규제" 공식은 항상 맞지 않음
  데이터셋 크기를 먼저 고려해야 함
```

### 3. Validation Gap이 성능 지표
```
양의 갭 (+): 언더피팅, 좋은 일반화 ✅
  ConvNeXt-Tiny: +0.07%p → LB 1위

음의 갭 (-): 오버피팅, 나쁜 일반화 ⚠️
  EfficientNet-B3: -0.39%p → LB 2위
  ConvNeXt-Small: -0.15%p → LB 3위
```

### 4. Early Stopping의 중요성
```
ConvNeXt-Tiny: Epoch 7 멈춤 (early)
  → 언더피팅 유지 → 96.25% ✅

ConvNeXt-Small: Epoch 18 멈춤 (late)
  → 약한 오버피팅 → 95.98% ❌

교훈:
  일찍 멈추는 것이 늦게 멈추는 것보다 나을 수 있음
```

### 5. Precision vs Recall 균형
```
균형잡힌 모델:
  ConvNeXt-Tiny:
    Precision: 96.67%
    Recall:    95.99%
    Diff: 0.68%p → H-Mean 96.25% ✅

불균형 모델:
  ConvNeXt-Small:
    Precision: 97.39%
    Recall:    94.83%
    Diff: 2.56%p → H-Mean 95.98% ❌

교훈:
  극단적인 Precision은 H-Mean을 해침
  균형이 최고 성능의 핵심
```

---

## 🔮 향후 개선 방안

### 1. Weight Decay 최적화 (최우선)
```python
제안: ConvNeXt-Small with optimized weight decay

Current:  wd=0.00012  → LB 95.98% ❌
Target:   wd=0.00009  → Expected 96.1-96.2%
          wd=0.0001   → Expected 96.15-96.25%

실험 계획:
  - wd in [0.00008, 0.00009, 0.0001]
  - 각 3회 반복 실험
  - Precision/Recall 균형 모니터링
```

### 2. Progressive Resolution 전략 수정
```
Current: 640px (0-3) → 960px (4-19)

Option A: 더 이른 전환
  640px (0-2) → 960px (3-19)
  → 960px 학습 시간 증가

Option B: 3단계 전환
  640px (0-3) → 800px (4-9) → 960px (10-19)
  → 점진적 해상도 증가

Option C: 고해상도만 사용
  960px (0-19)
  → 시작부터 최종 해상도
```

### 3. Learning Rate Schedule 조정
```python
Current:
  lr = 0.0004 (너무 보수적?)

Proposal:
  lr = 0.00043  # Tiny(0.00045)와 Baseline(0.0004) 중간
  + Warm-up 3 epochs
  + Cosine annealing with T_max=17 (3 warm-up 후)
```

### 4. Ensemble 전략 재고
```
기존 계획: ConvNeXt-Small 5-fold ensemble
새 계획: ConvNeXt-Tiny 5-fold ensemble (더 나은 선택)

이유:
  - Tiny가 단일 모델로도 최고 성능 (96.25%)
  - Small은 개선 여지 불확실
  - Tiny ensemble 예상 성능: 96.4-96.6%
```

### 5. Hybrid Approach
```
Option: Tiny + Small Ensemble
  - ConvNeXt-Tiny (3 folds):  각 96.25% 예상
  - ConvNeXt-Small (2 folds):  각 95.98-96.1% 예상
  - Soft voting ensemble
  - Expected: 96.3-96.5%

장점:
  - 다양성 확보
  - Tiny의 높은 Recall + Small의 높은 Precision
```

---

## 📊 최종 모델 순위 및 권장사항

### 성능 순위
```
🥇 1위: ConvNeXt-Tiny (28M)
   LB: 96.25%
   Val: 96.18%
   Gap: +0.07%p ✅
   → 최고 일반화 성능
   → 5-fold ensemble 1순위 권장 ⭐⭐⭐

🥈 2위: EfficientNet-B3 (12M)
   LB: 96.19%
   Val: 96.58%
   Gap: -0.39%p ⚠️
   → 오버피팅 문제
   → 개별 사용 가능, ensemble 비권장

🥉 3위: ConvNeXt-Small (50M)
   LB: 95.98%
   Val: 96.13%
   Gap: -0.15%p ⚠️
   → Weight decay 재조정 필요
   → 현재 상태로는 비권장 ❌
```

### 권장 전략
```
Short-term (즉시):
  ✅ ConvNeXt-Tiny 5-fold ensemble
     Expected LB: 96.4-96.6%
     Time: 6-8 hours
     Confidence: High

Mid-term (시간 있을 때):
  ⚠️ ConvNeXt-Small weight decay 최적화
     Test wd in [0.00008, 0.00009, 0.0001]
     If successful: 96.1-96.25% possible
     Time: 6-8 hours
     Confidence: Medium

Long-term (여유 있을 때):
  💡 Hybrid ensemble (Tiny + optimized Small)
     Expected LB: 96.5-96.7%
     Time: 12-16 hours
     Confidence: Medium-High
```

---

## 🎯 결론

### 핵심 발견 요약
1. **모델 크기 ≠ 성능**: 28M(Tiny) > 50M(Small) > 12M(B3)
2. **언더피팅 우위**: +0.07%p gap이 -0.39%p보다 훨씬 나음
3. **Regularization 역효과**: Weight decay 0.00012가 과도
4. **Precision/Recall 불균형**: 97.39%/94.83% → H-Mean 하락
5. **최적 선택**: ConvNeXt-Tiny가 현재 최고의 single model

### 최종 권장사항
```
✅ RECOMMENDED:
   ConvNeXt-Tiny (28M) 5-fold ensemble
   - 검증된 최고 성능 (LB 96.25%)
   - 양의 일반화 갭 (+0.07%p)
   - 균형잡힌 Precision/Recall
   - 예상 ensemble 성능: 96.4-96.6%

⚠️ OPTIONAL:
   ConvNeXt-Small weight decay 튜닝 실험
   - 현재 상태(wd=0.00012)는 비권장
   - 0.0001 이하로 재실험 필요
   - 성공시 Tiny와 hybrid ensemble 가능

❌ NOT RECOMMENDED:
   현재 ConvNeXt-Small(50M) 사용
   - 최하위 성능 (95.98%)
   - 재조정 없이는 사용 불가
```

### 교훈
**"데이터가 작을 때는 적당한 크기의 모델 + 적절한 규제가 큰 모델 + 강한 규제보다 낫다"**

이번 실험을 통해 우리는 더 큰 것이 항상 더 좋은 것은 아니며, 데이터셋 크기에 맞는 모델 선택과 적절한 regularization balance가 얼마나 중요한지 배웠습니다.

---

**Report End**

*다음 단계: ConvNeXt-Tiny 5-fold ensemble 진행 권장*
