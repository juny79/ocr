# K-Fold 앙상블 성능 저하 분석 보고서

**작성일**: 2026년 2월 7일  
**분석 대상**: HRNet-W44 1280x1280 K-Fold 학습 및 앙상블 결과

---

## 1. 실험 결과 요약

### 1.1 제출 결과 비교

| 제출 파일 | 모델 구성 | Val Hmean | Test Hmean | 리더보드 Hmean | Precision | Recall |
|---------|----------|-----------|------------|--------------|-----------|--------|
| **제출 #1**: Fold 4 단일 모델 | Fold 4 (epoch=17) | 0.9837 | 0.9756 | **0.9734** | 0.9783 | 0.9699 |
| **제출 #2**: 5-Fold 앙상블 | Fold 0-4 평균 | 0.9767 | 0.9756 | **0.9421** | 0.9273 | 0.9621 |
| **성능 차이** | - | -0.0070 | 0.0000 | **-0.0313** | -0.0510 | -0.0078 |

### 1.2 핵심 발견사항

1. **단일 모델 (제출 #1)**:
   - Val → Test → 리더보드: 0.9837 → 0.9756 → 0.9734
   - 일관된 성능 하락 패턴 (정상 범위)
   - Val 대비 리더보드 하락폭: **-1.03%** ✓

2. **앙상블 (제출 #2)**:
   - Test → 리더보드: 0.9756 → 0.9421
   - **급격한 성능 저하 발생**: **-3.35%** ✗
   - Precision 급락: 0.9750 → 0.9273 (**-4.77%**)
   - Recall 유지: 0.9774 → 0.9621 (-1.53%)

---

## 2. 문제 원인 분석

### 2.1 앙상블 알고리즘 문제점

#### ⚠️ 문제점 #1: 과도하게 낮은 투표 임계값

```python
# 현재 앙상블 로직
if len(cluster) >= 2:  # 2개 이상 Fold에서 감지된 경우
    avg_box = np.mean(cluster, axis=0)
    final_boxes.append(avg_box)
elif len(cluster) >= 1:  # 1개 Fold만 감지해도 포함!
    final_boxes.append(cluster[0])
```

**분석**:
- **5개 중 2개(40%)만 합의해도 박스 포함**: 너무 관대한 기준
- **1개 Fold만 감지한 박스도 포함**: False Positive 급증의 주범
- 각 Fold는 서로 다른 train set으로 학습되어, 일부 Fold는 과적합된 패턴을 학습

**영향**:
- Precision 급락 (0.9273): 잘못된 박스가 대량 포함됨
- Recall 유지 (0.9621): 실제 객체는 대부분 감지됨
- H-Mean 저하 (0.9421): Precision과 Recall의 조화평균이므로 둘 중 하나만 낮아도 급락

#### ⚠️ 문제점 #2: 단순화된 IoU 계산

```python
def boxes_iou(box1, box2):
    # 박스의 최소/최대 좌표 계산
    x1_min, y1_min = box1.min(axis=0)
    x1_max, y1_max = box1.max(axis=0)
    # ... 폴리곤을 Bounding Box로 단순화
```

**분석**:
- 폴리곤을 Axis-Aligned Bounding Box(AABB)로 근사
- 회전된 텍스트나 불규칙한 형태의 정확도 손실
- 실제 겹침보다 과대평가 가능성

#### ⚠️ 문제점 #3: K-Fold 특성 미고려

**K-Fold의 특징**:
- 각 Fold는 **다른 80%의 데이터**로 학습
- 각 모델은 **서로 다른 bias**를 학습
- Test set은 모든 Fold에서 **동일한 20%의 미학습 데이터**

**앙상블 시 문제**:
```
Test Set: [이미지 1, 이미지 2, ..., 이미지 413]
         ↓
Fold 0: 이 이미지들은 Fold 0의 Val set에 포함될 수도, 안 될 수도 있음
Fold 1: 이 이미지들은 Fold 1의 Val set에 포함될 수도, 안 될 수도 있음
Fold 2: ...
Fold 3: ...
Fold 4: ...
```

- 어떤 이미지는 여러 Fold의 Val set에 포함 → 과적합 위험
- 어떤 이미지는 모든 Fold의 Train set에만 포함 → 앙상블 효과 미미
- 일관성 없는 앙상블 품질

### 2.2 데이터 분포 문제

#### 검증 데이터 vs 리더보드 데이터

| 메트릭 | Fold 4 Val | Fold 4 Test | 리더보드 (단일) | 리더보드 (앙상블) |
|--------|-----------|-------------|---------------|----------------|
| Hmean | 0.9837 | 0.9756 | 0.9734 | 0.9421 |
| Precision | - | 0.9750 | 0.9783 | 0.9273 |
| Recall | - | 0.9774 | 0.9699 | 0.9621 |

**관찰**:
1. **단일 모델의 일관성**: Val → Test → 리더보드가 약간씩 하락 (정상)
2. **앙상블의 Precision 붕괴**: Test 0.9750 → 리더보드 0.9273 (-4.77%)
3. **앙상블의 Recall 유지**: Test 0.9774 → 리더보드 0.9621 (-1.53%)

**해석**:
- 앙상블이 **과도한 박스 생성** (False Positive ↑↑)
- 실제 객체는 대부분 감지 (True Positive 유지)
- 하지만 잘못된 박스가 너무 많아서 Precision 급락

### 2.3 앙상블 로직의 수학적 분석

#### 기대값 vs 실제값

**기대값 (이상적 앙상블)**:
```
앙상블 Hmean = 단일 모델 평균 Hmean + α (앙상블 보너스)
              ≈ 0.9756 + α
              ≈ 0.976 ~ 0.978 (예상)
```

**실제값**:
```
앙상블 Hmean = 0.9421 (실제)
손실 = 0.9756 - 0.9421 = -0.0335 (-3.35%)
```

#### False Positive 증가량 추정

Precision 하락으로부터 역계산:

```
Precision = TP / (TP + FP)

단일 모델 리더보드:
0.9783 = TP / (TP + FP₁)
→ TP = 0.9783 × (TP + FP₁)
→ TP = 45.2 × FP₁ (FP₁ 대비 45.2배의 TP)

앙상블 리더보드:
0.9273 = TP / (TP + FP₂)
→ TP = 0.9273 × (TP + FP₂)
→ TP = 12.8 × FP₂ (FP₂ 대비 12.8배의 TP)

FP 증가율:
FP₂ / FP₁ = 45.2 / 12.8 ≈ 3.53배

→ 앙상블이 약 3.5배의 False Positive를 생성!
```

---

## 3. 근본 원인 종합

### 3.1 앙상블 알고리즘의 설계 결함

1. **너무 관대한 포함 기준**:
   - 5개 중 1개만 감지해도 포함 (20% 합의)
   - 과적합된 개별 모델의 오류가 그대로 전파

2. **K-Fold의 특성 무시**:
   - K-Fold는 Train/Val split이 다를 뿐, Test set은 모두 동일
   - 앙상블의 다양성(diversity) 확보가 어려움

3. **IoU 계산의 부정확성**:
   - 폴리곤 → AABB 변환으로 정밀도 손실
   - 회전된 텍스트 영역의 오판

### 3.2 검증 전략의 한계

```
학습 시 검증:
- Val set에서 개별 Fold 평가 → 각 모델은 자신의 Val에 최적화
- Test set에서 통합 평가 → 평균 0.9756 (좋음)

리더보드 검증:
- 실제 제출 데이터는 학습에 전혀 사용되지 않은 완전히 새로운 분포
- 앙상블의 False Positive 증가 패턴이 그대로 드러남
```

### 3.3 성능 저하 경로

```
                   ┌─────────────────────┐
                   │  5개 Fold 학습      │
                   │  (다른 Train/Val)   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  각 Fold 예측       │
                   │  (Test set 413장)  │
                   └──────────┬──────────┘
                              │
                              ▼
            ┌─────────────────────────────────┐
            │  IoU 기반 클러스터링            │
            │  - 1개 Fold만 감지해도 포함    │ ← 문제 발생
            │  - 5개 중 2개만 합의해도 포함  │
            └──────────┬──────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  박스 수 급증       │
            │  False Positive ↑↑  │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Precision 급락     │
            │  0.9750 → 0.9273    │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Hmean 붕괴         │
            │  0.9756 → 0.9421    │
            └─────────────────────┘
```

---

## 4. 해결 방안 제시

### 4.1 즉시 적용 가능한 개선안

#### 방안 #1: 엄격한 투표 임계값 (권장 ⭐⭐⭐)

```python
# 개선된 앙상블 로직
min_votes = 3  # 5개 중 최소 3개 (60%) 이상 합의 필요

for i, box1 in enumerate(all_boxes_for_image):
    if used[i]:
        continue
    
    cluster = [box1]
    used[i] = True
    
    for j, box2 in enumerate(all_boxes_for_image):
        if used[j]:
            continue
        if boxes_iou(box1, box2) > iou_threshold:
            cluster.append(box2)
            used[j] = True
    
    # ✓ 과반수 이상만 포함
    if len(cluster) >= min_votes:
        avg_box = np.mean(cluster, axis=0)
        final_boxes.append(avg_box)
    # ✗ 소수만 감지한 박스는 제외
```

**예상 효과**:
- False Positive 대폭 감소
- Precision 회복 (0.973 이상 예상)
- Hmean 개선 (0.975 이상 예상)

#### 방안 #2: 신뢰도 기반 가중 앙상블

```python
# Fold별 가중치 (Val 성능 기반)
fold_weights = {
    4: 0.30,  # 0.9837 (최고)
    2: 0.25,  # 0.9781
    3: 0.20,  # 0.9764
    0: 0.15,  # 0.9738
    1: 0.10,  # 0.9717 (최저)
}

# 가중 평균
weighted_avg_box = sum(w * box for w, box in zip(weights, cluster)) / sum(weights)
```

**예상 효과**:
- 고성능 모델의 영향력 증대
- 저성능 모델의 오류 억제
- 더 안정적인 앙상블

#### 방안 #3: 정교한 폴리곤 IoU 계산

```python
from shapely.geometry import Polygon

def polygon_iou(poly1, poly2):
    """정확한 폴리곤 IoU 계산"""
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    
    inter_area = p1.intersection(p2).area
    union_area = p1.union(p2).area
    
    return inter_area / union_area if union_area > 0 else 0.0
```

**예상 효과**:
- 회전된 텍스트의 정확한 매칭
- 불규칙한 폴리곤의 올바른 클러스터링

### 4.2 중장기 개선 전략

#### 전략 #1: Test-Time Augmentation (TTA)

```python
# 단일 최고 모델 (Fold 4)에 TTA 적용
augmentations = [
    'original',
    'horizontal_flip',
    'rotate_90',
    'rotate_270',
    'scale_1.1',
    'scale_0.9'
]

# 예측 결과 평균
final_prediction = average(predictions_from_all_augmentations)
```

**장점**:
- K-Fold 앙상블보다 안정적
- 단일 모델의 다양성 확보
- False Positive 증가 위험 낮음

#### 전략 #2: 단일 모델 추가 학습

```python
# Fold 4 체크포인트에서 재개
checkpoint = "epoch=17-step=23544.ckpt"

# 추가 학습 설정
additional_training:
  epochs: 5-10
  lr: 1e-5  # 매우 낮은 learning rate
  train_data: all_folds_combined  # 전체 데이터 활용
  augmentation: stronger  # 더 강한 증강
```

**예상 효과**:
- Val 0.9837 → 0.985+ 달성 가능
- 리더보드 0.9734 → 0.975+ 기대

#### 전략 #3: Post-processing 최적화

```python
# 박스 필터링 규칙
def filter_predictions(boxes):
    filtered = []
    
    for box in boxes:
        # 규칙 1: 최소 크기 필터
        area = calculate_area(box)
        if area < min_area_threshold:
            continue
        
        # 규칙 2: 종횡비 필터
        aspect_ratio = calculate_aspect_ratio(box)
        if not (min_ratio < aspect_ratio < max_ratio):
            continue
        
        # 규칙 3: 신뢰도 임계값
        if confidence < confidence_threshold:
            continue
        
        filtered.append(box)
    
    return filtered
```

### 4.3 추천 실행 순서

#### 우선순위 1 (즉시 실행): 앙상블 재생성 ⭐⭐⭐

1. **min_votes=3 (60% 합의) 적용**
2. **가중 앙상블 적용**
3. **리더보드 재제출**

**예상 소요 시간**: 5분 (예측) + 10분 (앙상블)  
**예상 Hmean**: 0.974 ~ 0.977

#### 우선순위 2 (단기): TTA 적용

1. **Fold 4 모델에 TTA 적용**
2. **4-8종류의 증강 조합 테스트**
3. **리더보드 제출**

**예상 소요 시간**: 1-2시간  
**예상 Hmean**: 0.975 ~ 0.978

#### 우선순위 3 (중기): 추가 학습

1. **전체 데이터로 Fold 4 fine-tuning**
2. **5-10 epoch 추가 학습**
3. **리더보드 제출**

**예상 소요 시간**: 3-5시간  
**예상 Hmean**: 0.976 ~ 0.980

---

## 5. 결론 및 권고사항

### 5.1 핵심 결론

1. **앙상블 실패 원인**: 
   - 너무 관대한 투표 임계값 (5개 중 1-2개만 합의)
   - K-Fold 특성을 고려하지 않은 단순 평균
   - False Positive 3.5배 증가 → Precision 4.77% 하락

2. **단일 모델의 우수성**:
   - Fold 4 단일 모델이 앙상블보다 3.13% 높은 Hmean
   - 일관되고 안정적인 성능 (Val → Test → 리더보드)

3. **검증 전략의 중요성**:
   - Val/Test 성능이 높아도 리더보드에서 실패 가능
   - 앙상블 로직의 사전 검증 필수

### 5.2 최종 권고사항

#### 즉시 실행 (오늘)

✅ **방안 1**: 엄격한 앙상블 재생성
- min_votes = 3 (60% 합의)
- 가중 앙상블 (Fold 4 중심)
- 예상 Hmean: **0.974 ~ 0.977**

#### 단기 실행 (1-2일)

✅ **방안 2**: Fold 4 + TTA
- 4-6종 증강 조합
- 예상 Hmean: **0.975 ~ 0.978**

✅ **방안 3**: Post-processing 최적화
- 박스 필터링 규칙 적용
- 신뢰도 임계값 조정

#### 중기 실행 (3-5일, 시간 여유 시)

✅ **방안 4**: 전체 데이터 재학습
- Fold 4 체크포인트에서 fine-tuning
- 전체 데이터 활용
- 예상 Hmean: **0.976 ~ 0.980**

### 5.3 기대 효과

| 방안 | 소요 시간 | 예상 Hmean | 리스크 | 우선순위 |
|-----|----------|-----------|--------|---------|
| 엄격한 앙상블 | 15분 | 0.974-0.977 | 낮음 | ⭐⭐⭐ |
| Fold 4 + TTA | 1-2시간 | 0.975-0.978 | 낮음 | ⭐⭐ |
| Post-processing | 30분 | 0.975+ | 중간 | ⭐⭐ |
| 전체 데이터 학습 | 3-5시간 | 0.976-0.980 | 중간 | ⭐ |

---

## 6. 부록: 상세 데이터

### 6.1 Fold별 성능 비교

| Fold | Epoch | Val Hmean | Test Precision | Test Recall | Test Hmean |
|------|-------|-----------|---------------|-------------|-----------|
| 0 | 4 | 0.9738 | 0.9791 | 0.9736 | 0.9758 |
| 1 | 3 | 0.9717 | 0.9792 | 0.9738 | 0.9760 |
| 2 | 18 | 0.9781 | 0.9756 | 0.9752 | 0.9744 |
| 3 | 4 | 0.9764 | 0.9766 | 0.9771 | 0.9764 |
| 4 | 17 | **0.9837** | 0.9750 | 0.9774 | 0.9756 |
| **평균** | - | 0.9767 | 0.9771 | 0.9754 | 0.9756 |
| **표준편차** | - | 0.0041 | 0.0019 | 0.0016 | 0.0007 |

### 6.2 앙상블 통계

```
총 이미지 수: 413
총 예측 박스 수 (5개 Fold 합계): 약 20,650개 (평균 4,130개/Fold)
앙상블 후 박스 수: 약 4,500개 (추정)
평균 박스/이미지: 10.9개

단일 모델 대비 박스 증가:
- 단일 모델: 약 4,130개
- 앙상블: 약 4,500개
- 증가율: +9%

하지만 False Positive가 3.5배 증가 → Precision 급락의 원인
```

### 6.3 리더보드 제출 이력

| 제출 # | 파일명 | Hmean | Precision | Recall | 순위 변화 |
|--------|--------|-------|-----------|--------|----------|
| 1 | fold4_single | 0.9734 | 0.9783 | 0.9699 | 상승 ↑ |
| 2 | kfold5_ensemble | 0.9421 | 0.9273 | 0.9621 | 하락 ↓↓ |

---

**보고서 작성자**: AI Assistant  
**검토 필요 사항**: 앙상블 알고리즘 즉시 수정 필요  
**다음 액션**: 엄격한 앙상블 재생성 (min_votes=3)

