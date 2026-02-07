# 리더보드 제출 결과 분석 및 추가 개선 방안

**일시**: 2026년 2월 7일  
**제출 파일**: hrnet_w44_kfold5_ensemble_improved.csv  
**현재 최고 성능**: H-Mean 0.9755

---

## 🎉 성공적인 개선 달성!

### 제출 결과 비교

| 제출 # | 파일 | 모델 구성 | Hmean | Precision | Recall | 개선폭 |
|--------|------|----------|-------|-----------|--------|--------|
| 1 | fold4_single | Fold 4 단일 | 0.9734 | 0.9783 | 0.9699 | - |
| 2 | kfold5_ensemble | 기존 앙상블 (1개 투표) | 0.9421 | 0.9273 | 0.9621 | **-0.0313** ❌ |
| 3 | **kfold5_ensemble_improved** | **개선 앙상블 (3개 투표)** | **0.9755** | **0.9833** | **0.9688** | **+0.0021** ✅ |

### 핵심 성과

1. **기존 앙상블 대비**: +0.0334 (3.34% 향상) 🚀
2. **단일 모델 대비**: +0.0021 (0.21% 향상) ✅
3. **False Positive 제거 성공**: Precision 0.9273 → 0.9833 (+5.6%)
4. **현재 순위**: 최고 성능 달성!

---

## 📊 상세 분석

### 1. 메트릭 분해 분석

```
개선 앙상블 결과:
- H-Mean: 0.9755 = 2 × (0.9833 × 0.9688) / (0.9833 + 0.9688)
- Precision: 0.9833 (매우 높음 ⭐)
- Recall: 0.9688 (약간 낮음 ⚠️)
```

**관찰 사항**:
- **Precision 0.9833**: False Positive가 거의 없음 (엄격한 필터링 성공)
- **Recall 0.9688**: 일부 True Positive 누락 (너무 엄격한 기준?)
- **불균형**: Precision과 Recall의 차이 1.45%

### 2. 각 제출의 특징

#### 제출 #1: Fold 4 단일 모델
```
Precision: 0.9783 (균형)
Recall:    0.9699 (균형)
차이:      0.84%  → 균형잡힌 성능
```

#### 제출 #2: 기존 앙상블 (실패)
```
Precision: 0.9273 (낮음 - False Positive 과다)
Recall:    0.9621 (상대적으로 높음)
차이:      3.48%  → False Positive 문제
```

#### 제출 #3: 개선 앙상블 (성공)
```
Precision: 0.9833 (매우 높음 - False Positive 최소화)
Recall:    0.9688 (약간 낮음 - 일부 True Positive 누락)
차이:      1.45%  → 균형 개선 가능
```

### 3. 성능 개선 잠재력 분석

현재 Hmean이 0.9755인데, 이론적 최대값을 계산하면:

```python
# 시나리오 1: Recall을 Precision 수준으로 상향
if Recall = 0.9833:
    Hmean = 2 × (0.9833 × 0.9833) / (0.9833 + 0.9833)
    Hmean = 0.9833 (이론적 최대)

# 시나리오 2: Precision 유지, Recall만 +1%
if Recall = 0.9788 (현재 +1%):
    Hmean = 2 × (0.9833 × 0.9788) / (0.9833 + 0.9788)
    Hmean = 0.9810 (+0.0055, +0.55%)

# 시나리오 3: 균형 개선 (둘 다 약간 조정)
if Precision = 0.9820, Recall = 0.9720:
    Hmean = 2 × (0.9820 × 0.9720) / (0.9820 + 0.9720)
    Hmean = 0.9769 (+0.0014, +0.14%)
```

**결론**: Recall을 1% 올리면 Hmean이 0.981까지 가능!

---

## 🚀 추가 개선 방안

### 방안 #1: 적응형 투표 임계값 (권장 ⭐⭐⭐)

**현재 문제**: 모든 이미지에 동일한 임계값(3개) 적용
**개선안**: Fold 간 합의도에 따라 임계값 조정

```python
def adaptive_voting(cluster, iou_scores):
    """
    Fold 간 IoU가 높으면 → 낮은 임계값 허용 (2개)
    Fold 간 IoU가 낮으면 → 높은 임계값 유지 (3개)
    """
    vote_count = len(cluster)
    avg_iou = np.mean(iou_scores)
    
    # IoU 기반 임계값 조정
    if avg_iou > 0.7:  # 매우 높은 합의
        threshold = 2  # 2개 Fold만으로도 충분
    elif avg_iou > 0.6:  # 높은 합의
        threshold = 2.5  # 가중치 고려
    else:  # 낮은 합의
        threshold = 3  # 엄격하게 유지
    
    return vote_count >= threshold
```

**예상 효과**:
- Recall 향상: 0.9688 → 0.9720 (+0.32%)
- Precision 유지: 0.9833 → 0.9820 (-0.13%)
- **Hmean 예상**: 0.9769 (+0.14%)

### 방안 #2: 신뢰도 기반 2단계 필터링

**개념**: 투표 수 + 모델 신뢰도 결합

```python
def confidence_based_voting(cluster, fold_confidences):
    """
    3개 미만 투표라도 신뢰도가 매우 높으면 포함
    3개 이상 투표라도 신뢰도가 낮으면 제외
    """
    vote_count = len(cluster)
    avg_confidence = np.mean([fold_confidences[c['fold_idx']] for c in cluster])
    
    if vote_count >= 3:
        return True  # 기본 통과
    elif vote_count == 2 and avg_confidence > 0.95:
        return True  # 높은 신뢰도 → 2개 투표도 허용
    elif vote_count >= 4 and avg_confidence < 0.7:
        return False  # 낮은 신뢰도 → 많은 투표도 거부
    else:
        return vote_count >= 3
```

**예상 효과**:
- Recall 향상: 0.9688 → 0.9710 (+0.22%)
- Precision 약간 하락: 0.9833 → 0.9825 (-0.08%)
- **Hmean 예상**: 0.9767 (+0.12%)

### 방안 #3: Test-Time Augmentation (TTA) 적용 (권장 ⭐⭐⭐)

**개념**: Fold 4 최고 모델에 증강 적용

```python
augmentations = [
    {'name': 'original', 'flip': False, 'rotate': 0},
    {'name': 'hflip', 'flip': True, 'rotate': 0},
    {'name': 'rotate_90', 'flip': False, 'rotate': 90},
    {'name': 'rotate_270', 'flip': False, 'rotate': 270},
]

# 각 증강으로 예측 후 평균
predictions_tta = []
for aug in augmentations:
    pred = model.predict(image, augmentation=aug)
    pred = inverse_transform(pred, aug)  # 역변환
    predictions_tta.append(pred)

final_prediction = weighted_average(predictions_tta)
```

**장점**:
- Recall 향상 (다양한 각도에서 감지)
- Precision 유지 (일관성 있는 박스만 유지)
- 앙상블보다 안정적

**예상 효과**:
- Recall 향상: 0.9688 → 0.9740 (+0.52%)
- Precision 유지: 0.9833 → 0.9820 (-0.13%)
- **Hmean 예상**: 0.9779 (+0.24%)

**소요 시간**: 1-2시간

### 방안 #4: 박스 크기 기반 필터링

**관찰**: 매우 작거나 큰 박스는 오탐일 가능성 높음

```python
def filter_by_box_size(boxes, image_size):
    """박스 크기 기반 필터링"""
    filtered = []
    
    for box in boxes:
        area = calculate_polygon_area(box)
        image_area = image_size[0] * image_size[1]
        ratio = area / image_area
        
        # 너무 작은 박스 제거 (노이즈)
        if ratio < 0.0001:  # 0.01% 미만
            continue
        
        # 너무 큰 박스 제거 (전체 이미지)
        if ratio > 0.5:  # 50% 이상
            continue
        
        # 종횡비 필터 (텍스트는 대부분 가로로 김)
        width, height = calculate_bbox(box)
        aspect_ratio = width / height if height > 0 else 0
        
        # 너무 정사각형이거나 너무 긴 박스 제거
        if aspect_ratio < 0.2 or aspect_ratio > 20:
            continue
        
        filtered.append(box)
    
    return filtered
```

**예상 효과**:
- Recall 약간 하락: 0.9688 → 0.9680 (-0.08%)
- Precision 향상: 0.9833 → 0.9845 (+0.12%)
- **Hmean 예상**: 0.9762 (+0.07%)

### 방안 #5: 가중치 재조정

**현재 가중치**:
```python
fold_weights = {
    4: 0.30,  # Val 0.9837
    2: 0.25,  # Val 0.9781
    3: 0.20,  # Val 0.9764
    0: 0.15,  # Val 0.9738
    1: 0.10,  # Val 0.9717
}
```

**제안 가중치** (Fold 4 중심):
```python
fold_weights_aggressive = {
    4: 0.40,  # Val 0.9837 (최고 모델 더 강조)
    2: 0.25,  # Val 0.9781
    3: 0.20,  # Val 0.9764
    0: 0.10,  # Val 0.9738
    1: 0.05,  # Val 0.9717 (최저 모델 약화)
}
```

**예상 효과**:
- 고성능 모델의 영향력 증대
- 저성능 모델의 노이즈 감소
- **Hmean 예상**: 0.9760 (+0.05%)

---

## 🎯 추천 실행 계획

### 우선순위 1: TTA 적용 (즉시 실행) ⭐⭐⭐

**이유**:
- 가장 큰 성능 향상 기대 (+0.24%)
- 단일 최고 모델(Fold 4) 활용
- 앙상블보다 안정적

**예상 결과**:
- Hmean: 0.9779
- Precision: 0.9820
- Recall: 0.9740

**실행 계획**:
1. Fold 4 체크포인트로 TTA 스크립트 작성
2. 4가지 증강 (원본, 좌우반전, 90도, 270도 회전)
3. 예측 결과 역변환 및 평균
4. 리더보드 제출

**소요 시간**: 1-2시간

### 우선순위 2: 적응형 투표 임계값 (단기) ⭐⭐

**이유**:
- 현재 앙상블 프레임워크 활용
- 빠른 구현 가능 (30분)
- Recall 개선 효과

**예상 결과**:
- Hmean: 0.9769
- Precision: 0.9820
- Recall: 0.9720

**실행 계획**:
1. 앙상블 스크립트에 IoU 기반 적응형 임계값 추가
2. 재실행 및 제출

**소요 시간**: 30분

### 우선순위 3: 신뢰도 기반 필터링 (중기) ⭐

**이유**:
- 추가 정보 활용 (모델 신뢰도)
- 더 정교한 필터링

**예상 결과**:
- Hmean: 0.9767
- Precision: 0.9825
- Recall: 0.9710

**실행 계획**:
1. 모델 출력에서 신뢰도 점수 추출
2. 신뢰도 기반 투표 로직 구현
3. 리더보드 제출

**소요 시간**: 1-2시간

---

## 📈 성능 향상 로드맵

### 단계별 목표

```
현재:     Hmean 0.9755 ✓

단기 목표: Hmean 0.9770 (+0.15%, 1-2일)
├─ 적응형 투표 임계값: +0.0014
└─ 가중치 재조정: +0.0005

중기 목표: Hmean 0.9780 (+0.25%, 3-5일)
├─ TTA 적용: +0.0024
├─ 신뢰도 기반 필터링: +0.0012
└─ 박스 크기 필터링: +0.0007

장기 목표: Hmean 0.9800+ (+0.45%, 1-2주)
├─ 전체 데이터 재학습
├─ 추가 증강 기법
└─ 앙상블 + TTA 결합
```

---

## 🎓 현재까지 배운 교훈

### 성공 요인

1. ✅ **엄격한 투표 임계값**: 60% 합의 (5개 중 3개)
2. ✅ **가중 앙상블**: 고성능 모델 중심
3. ✅ **정교한 IoU**: Shapely 기반 폴리곤 계산
4. ✅ **체계적 분석**: 수치 기반 의사결정

### 개선 포인트

1. 🔸 **Recall 향상 필요**: 현재 0.9688 → 목표 0.9740
2. 🔸 **Precision 유지**: 현재 0.9833 (매우 좋음)
3. 🔸 **균형 조정**: Precision-Recall 차이 최소화

---

## 📊 비교 요약표

| 방안 | 소요 시간 | Hmean 예상 | Precision | Recall | 구현 난이도 | 우선순위 |
|-----|----------|-----------|-----------|--------|-----------|---------|
| **TTA** | 1-2시간 | **0.9779** | 0.9820 | 0.9740 | 중간 | ⭐⭐⭐ |
| 적응형 투표 | 30분 | 0.9769 | 0.9820 | 0.9720 | 쉬움 | ⭐⭐ |
| 신뢰도 필터링 | 1-2시간 | 0.9767 | 0.9825 | 0.9710 | 중간 | ⭐⭐ |
| 박스 크기 필터 | 30분 | 0.9762 | 0.9845 | 0.9680 | 쉬움 | ⭐ |
| 가중치 재조정 | 15분 | 0.9760 | 0.9820 | 0.9700 | 매우 쉬움 | ⭐ |

---

## 🎯 최종 권고사항

### 즉시 실행 (추천)

✅ **Option A: TTA 적용 (Fold 4)**
- 예상 Hmean: **0.9779** (+0.0024)
- 가장 높은 성능 향상
- 단일 모델의 안정성 + 앙상블 효과

✅ **Option B: 적응형 투표 + 가중치 재조정**
- 예상 Hmean: **0.9769** (+0.0014)
- 빠른 구현 (45분)
- 현재 프레임워크 활용

### 보수적 전략

현재 0.9755가 이미 매우 우수한 성능이므로:
- 추가 실험은 선택 사항
- 현재 성능 유지도 충분히 경쟁력 있음
- 시간 대비 효과 고려 필요

### 공격적 전략

0.9800+ 목표:
1. TTA 적용 (Fold 4) → 0.9779
2. 적응형 투표 추가 → 0.9785
3. 전체 데이터 재학습 (Fold 4 기반) → 0.9800+

---

## 📌 결론

**현재 상태**: 
- Hmean 0.9755 (최고 수준 ✅)
- 기존 앙상블 대비 +3.34% 개선
- 단일 모델 대비 +0.21% 개선

**추가 개선 잠재력**:
- TTA 적용 시: +0.24% → **0.9779**
- 적응형 투표: +0.14% → **0.9769**
- 신뢰도 필터링: +0.12% → **0.9767**

**최종 추천**: 
1. **즉시**: TTA 적용 (1-2시간, Hmean 0.9779 목표)
2. **단기**: 적응형 투표 (30분, Hmean 0.9769 목표)
3. **관찰**: 현재 성능 유지도 충분히 경쟁력 있음

**다음 액션**: TTA 스크립트 작성 여부 결정

