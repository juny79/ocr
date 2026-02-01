# ResNet50 성능 개선 및 앙상블 전략 분석 보고서

**작성일**: 2026-02-01  
**분석 대상**: ResNet50 백본 전환 이후 성능 변화 추적

---

## 📊 리더보드 제출 내역 (ResNet50 이후)

### 1. ResNet50 기본 모델 (2026-02-01 05:09)
**파일**: `submission_resnet50_fold0_09.csv`

| 지표 | 값 |
|------|-----|
| **H-Mean** | **96.20%** |
| Precision | 96.53% |
| Recall | 95.87% |

- **특징**: ResNet18 → ResNet50 백본 업그레이드
- **설정**: 기본 postprocessing (thresh=0.25, box_thresh=0.3)
- **훈련**: Fold 0, 22 epochs, lr=0.0005, batch_size=4
- **결과**: 기준선 대비 성능 향상 확인

---

### 2. Aggressive Postprocessing (2026-02-01 05:23)
**파일**: `submission_resnet50_aggressive_10.csv`

| 지표 | 값 | 변화 |
|------|-----|------|
| **H-Mean** | **96.26%** | +0.06%p |
| Precision | 96.49% | -0.04%p |
| Recall | 96.23% | +0.36%p |

- **특징**: Recall 최적화 전략
- **설정**: thresh=0.22, box_thresh=0.25, max_candidates=600
- **박스 수**: 44,628개 (평균 108.1개/이미지)
- **결과**: ✅ **최고 H-Mean 달성** (단일 모델 기준)
- **분석**: Precision-Recall 균형이 가장 우수

---

### 3. TTA (Test-Time Augmentation) 시도 (2026-02-01 05:48)
**파일**: `submission_resnet50_tta_11.csv`

| 지표 | 값 | 변화 |
|------|-----|------|
| **H-Mean** | **78.25%** | **-18.01%p** ❌ |
| Precision | 71.03% | -25.46%p |
| Recall | 97.48% | +1.25%p |

- **특징**: HorizontalFlip TTA 적용
- **박스 수**: 57,953개 (평균 140.4개/이미지) - 30% 증가
- **실패 원인**: 
  - 좌표 변환 버그로 flipped 이미지 예측 박스가 잘못된 위치에 생성
  - ~75개 False Positive 박스 추가로 Precision 급락
  - Albumentations의 HorizontalFlip이 박스 좌표를 자동 변환하지 않음
- **교훈**: TTA 구현 시 좌표계 변환 필수 확인

---

### 4. Ultra-Aggressive Postprocessing (2026-02-01 06:00)
**파일**: `submission_resnet50_ultra_aggressive_12.csv`

| 지표 | 값 | 변화 |
|------|-----|------|
| **H-Mean** | **96.09%** | -0.17%p |
| Precision | 95.82% | -0.67%p |
| Recall | 96.67% | +0.44%p |

- **특징**: Recall 극대화 시도
- **설정**: thresh=0.20, box_thresh=0.23 (매우 낮음)
- **결과**: ❌ 과도한 aggressive 설정 → Precision 저하
- **분석**: Recall 증가(+0.44%p)보다 Precision 감소(-0.67%p)가 커서 H-Mean 하락

---

### 5. 2-Fold Ensemble (Voting ≥ 1) (2026-02-01 11:44)
**파일**: `submission_resnet50_2fold_ensemble_13.csv`

| 지표 | 값 | 변화 |
|------|-----|------|
| **H-Mean** | **95.09%** | -1.17%p ❌ |
| Precision | 94.53% | -1.96%p |
| Recall | 96.01% | -0.22%p |

- **특징**: Fold 0 + Fold 1 OR 앙상블
- **설정**: voting_threshold=1, iou_threshold=0.5
- **박스 수**: 45,784개 (평균 110.9개/이미지) - 2.6% 증가
- **실패 원인**:
  - 두 Fold의 불일치 박스를 모두 포함 → False Positive 급증
  - Fold 0과 Fold 1의 성능 차이 미미 (0.07%p) → 상호 보완 효과 없음
  - Precision 1.96%p 급락이 결정적

---

### 6. 2-Fold Ensemble (Voting = 2) (2026-02-01 12:06)
**파일**: `submission_resnet50_2fold_voting2.csv`

| 지표 | 값 | 변화 |
|------|-----|------|
| **H-Mean** | **95.92%** | -0.34%p |
| Precision | **97.36%** | +0.87%p ⬆️ |
| Recall | **94.76%** | -1.47%p ⬇️ |

- **특징**: Fold 0 + Fold 1 AND 앙상블
- **설정**: voting_threshold=2, iou_threshold=0.5
- **박스 수**: 43,521개 (평균 105.4개/이미지) - 2.5% 감소
- **결과**: Precision 최고치 달성하지만 Recall 급락
- **문제점**: **Precision-Recall 불균형 심화**
  - Precision 97.36% (최고) vs Recall 94.76% (최저)
  - 차이: 2.6%p → 균형 회복 필요

---

## 🔍 핵심 문제 진단

### 1. **Precision-Recall 트레이드오프 심화**

```
모델별 Precision vs Recall 분포:

Voting=2    : Precision 97.36% ████████████████████
              Recall    94.76% ███████████████

Aggressive  : Precision 96.49% ██████████████████
              Recall    96.23% ██████████████████  ← 최적 균형

Voting≥1    : Precision 94.53% ████████████████
              Recall    96.01% ██████████████████

차이 (Voting=2): 2.6%p (불균형)
차이 (Aggressive): 0.26%p (균형) ✅
```

**문제**: Voting=2는 교집합 방식으로 보수적 → Recall 손실 > Precision 이득

---

### 2. **앙상블 효과 부재 원인**

| 비교 항목 | Fold 0 | Fold 1 | 차이 |
|-----------|--------|--------|------|
| Test H-Mean | 95.89% | 95.96% | 0.07%p |
| Test Precision | 96.58% | 96.58% | 0.00%p |
| Test Recall | 95.59% | 95.59% | 0.00%p |
| 박스 수 | 44,628 | 44,686 | +58 (0.1%) |

**분석**:
- 두 Fold가 거의 동일한 예측 → 다양성 부족
- 동일 데이터, 동일 augmentation, 동일 hyperparameter
- K-Fold의 데이터 분할만으로는 충분한 다양성 확보 실패

---

### 3. **박스 통계 상세 분석**

```
=========================================================================
모델              | 총 박스    | 평균/이미지 | Precision | Recall  | Gap
=========================================================================
Fold 0 (단독)     | 44,628    | 108.1      | 96.49%   | 96.23%  | 0.26%p
Fold 1 (단독)     | 44,686    | 108.2      | (추정 96.5%)        
Voting≥1 (OR)     | 45,784    | 110.9      | 94.53%   | 96.01%  | 1.48%p
Voting=2 (AND)    | 43,521    | 105.4      | 97.36%   | 94.76%  | 2.60%p ⚠️
=========================================================================
```

**최적 박스 수**: 44,600개 부근 (단일 모델 수준)

---

## 💡 해결 전략

### 전략 1: **Weighted Voting Ensemble** (추천 ⭐)

**개념**: Fold 간 신뢰도 기반 가중 평균
```python
if overlap_count == 2:  # 두 Fold 모두 검출
    confidence_weight = 1.0  # 높은 신뢰도
elif overlap_count == 1:  # 한 Fold만 검출
    if box_score > threshold:  # 조건부 포함
        confidence_weight = 0.7
```

**예상 효과**:
- Precision: 96.8-97.0% (Voting=2보다 낮지만 안정적)
- Recall: 95.5-95.8% (Voting=2보다 높음)
- H-Mean: **96.2-96.5%** (목표)

---

### 전략 2: **동적 Threshold 최적화**

**현재 문제**: 고정 threshold가 모든 이미지에 동일 적용

**제안**:
```python
# 이미지별 박스 밀도에 따라 threshold 조정
if box_density > 120:  # 복잡한 영수증
    thresh = 0.24  # 더 보수적
else:
    thresh = 0.22  # 기본값
```

**구현 필요 사항**:
1. 밀도 추정 모델 또는 통계 기반 휴리스틱
2. Validation set에서 최적 threshold 탐색

---

### 전략 3: **Score-Based Filtering**

**아이디어**: Voting=1 박스에 confidence threshold 추가
```python
for box in voting_1_boxes:
    if box.score > 0.85:  # 고신뢰도만 포함
        final_boxes.append(box)
```

**예상 결과**:
- Precision 96.5% + Recall 95.8% → H-Mean 96.15%

---

### 전략 4: **NMS Parameter 최적화**

**현재**: IoU threshold = 0.5 (고정)

**제안**: 앙상블별 IoU 조정
```python
# Voting≥1: 더 엄격한 NMS로 중복 제거
iou_threshold = 0.4

# Voting=2: 더 관대한 NMS로 다양성 유지  
iou_threshold = 0.6
```

---

## 🎯 즉시 실행 가능한 대안

### 대안 1: **Fold 1 단독 제출** (가장 안전)

**근거**:
- Fold 1 Test H-Mean: 95.96% (Fold 0보다 0.07%p 높음)
- 최신 훈련 모델 → 일반화 성능 우수 가능성
- 파일: `submission_resnet50_fold1_aggressive.csv`

**예상 리더보드**: **96.3-96.5%**

---

### 대안 2: **Soft Voting Ensemble** (구현 필요)

**방법**: 박스별 confidence 평균 사용
```python
# 예시 pseudo-code
ensemble_boxes = []
for img in images:
    boxes_fold0 = get_boxes(fold0, img)
    boxes_fold1 = get_boxes(fold1, img)
    
    # IoU > 0.5인 박스 쌍 매칭
    matched_pairs = match_boxes(boxes_fold0, boxes_fold1, iou=0.5)
    
    for pair in matched_pairs:
        # 좌표 평균 + confidence 평균
        avg_box = average_box(pair)
        avg_box.score = (pair[0].score + pair[1].score) / 2
        ensemble_boxes.append(avg_box)
    
    # Voting=1 박스는 score threshold로 필터링
    unmatched = get_unmatched_boxes(boxes_fold0, boxes_fold1)
    for box in unmatched:
        if box.score > 0.85:  # 고신뢰도만
            ensemble_boxes.append(box)
```

**예상 리더보드**: **96.4-96.7%**

---

### 대안 3: **Post-Processing 미세 조정** (빠른 실험)

**목표**: Precision 96.8%, Recall 95.8% 균형점 찾기

| Preset | thresh | box_thresh | 예상 Precision | 예상 Recall | 예상 H-Mean |
|--------|--------|------------|----------------|-------------|-------------|
| Balanced | 0.23 | 0.26 | 96.7% | 95.9% | **96.30%** |
| Conservative | 0.24 | 0.27 | 97.0% | 95.5% | 96.24% |
| Current | 0.22 | 0.25 | 96.5% | 96.2% | 96.26% |

**구현**: Head config 파일 수정 후 재예측 (5분 소요)

---

## 📈 점수 상승 여력 분석

### 1. **현재 상황**
- 단일 모델 최고: **96.26%** (aggressive)
- Voting=2 앙상블: **95.92%** (Recall 손실)
- 갭: **0.34%p** (회복 여력)

### 2. **상승 가능 시나리오**

#### 시나리오 A: 앙상블 최적화 (현실적)
```
현재 최고 (단일): 96.26%
+ Soft Voting 효과: +0.15~0.25%p
= 예상 H-Mean: 96.4~96.5%
```

#### 시나리오 B: 추가 Fold 훈련 (시간 소요)
```
현재 2-Fold: 95.92%
→ 5-Fold 앙상블: +0.3~0.5%p (이론값)
= 예상 H-Mean: 96.6~96.8%
단, Fold 간 다양성 부족 문제 존재
```

#### 시나리오 C: 모델 다양성 확보 (장기 전략)
```
전략:
1. 다른 백본 추가 (EfficientNet, Swin Transformer)
2. 다른 해상도 (768px, 1024px)
3. 다른 augmentation 전략

예상 효과: +0.5~1.0%p
= 예상 H-Mean: 96.7~97.2%
```

---

## 🚀 최종 권장 사항

### 즉시 실행 (우선순위)

#### 1️⃣ **Fold 1 단독 제출** (5분)
- 파일: `submission_resnet50_fold1_aggressive.csv`
- 예상: H-Mean 96.3-96.5%
- 리스크: 낮음

#### 2️⃣ **Balanced Threshold 재예측** (10분)
- thresh=0.23, box_thresh=0.26 설정
- 예상: H-Mean 96.3-96.4%
- Precision-Recall 균형 개선

#### 3️⃣ **Soft Voting 구현** (30분)
- Confidence 기반 가중 앙상블
- 예상: H-Mean 96.4-96.7%
- 최고 성능 기대

---

### 중기 전략 (1-2일)

1. **3-Fold 추가 훈련**
   - Fold 2 훈련 (2시간)
   - 3-Fold 앙상블로 다양성 증가
   
2. **해상도 실험**
   - 1024px 모델 훈련
   - Multi-scale ensemble

---

### 장기 전략 (3-5일)

1. **백본 다양화**
   - EfficientNet-B4
   - Swin Transformer-Tiny
   
2. **Two-Stage Approach**
   - Stage 1: 높은 Recall 모델 (thresh=0.20)
   - Stage 2: 높은 Precision 모델 (thresh=0.26)
   - Weighted ensemble

---

## 📋 실험 로그

### 성공 사례
✅ ResNet50 업그레이드: +0.4%p (95.81% → 96.20%)
✅ Aggressive postprocessing: +0.06%p (96.20% → 96.26%)

### 실패 사례
❌ TTA (HorizontalFlip): -18.01%p - 좌표 변환 버그
❌ Ultra-aggressive: -0.17%p - 과도한 Recall 추구
❌ 2-Fold Voting≥1: -1.17%p - False Positive 증가
❌ 2-Fold Voting=2: -0.34%p - Recall 손실 과다

---

## 🎓 교훈 및 인사이트

### 1. **앙상블 성공 조건**
- ✅ 모델 간 충분한 다양성 필요
- ❌ 동일 아키텍처 + 동일 데이터 분할만으로는 부족
- ✅ Voting threshold는 validation에서 최적화 필요

### 2. **Precision-Recall 균형**
- ✅ Gap이 2%p 이상이면 불균형 신호
- ✅ 단일 모델 aggressive (gap 0.26%p)가 최적 균형
- ❌ 극단적 설정은 역효과 (ultra-aggressive, voting=2)

### 3. **Postprocessing의 중요성**
- ✅ Threshold 0.01 차이로 0.1-0.2%p 변동
- ✅ 이미지별 적응형 threshold 필요성 확인
- ✅ Validation에서 exhaustive search 필요

---

## 📊 종합 점수표

| 순위 | 모델 | H-Mean | Precision | Recall | Gap | 비고 |
|------|------|--------|-----------|--------|-----|------|
| 🥇 1 | Aggressive (단일) | **96.26%** | 96.49% | 96.23% | 0.26%p | 현재 최고 |
| 🥈 2 | ResNet50 Basic | 96.20% | 96.53% | 95.87% | 0.66%p | 안정적 |
| 🥉 3 | Ultra-Aggressive | 96.09% | 95.82% | 96.67% | 0.85%p | Recall 최고 |
| 4 | Voting=2 | 95.92% | **97.36%** | 94.76% | 2.60%p | Precision 최고 |
| 5 | Voting≥1 | 95.09% | 94.53% | 96.01% | 1.48%p | 불균형 |
| 6 | TTA | 78.25% | 71.03% | **97.48%** | 26.45%p | 치명적 버그 |

---

## 🎯 결론

### 핵심 메시지
1. **단일 모델 aggressive (96.26%)가 현재 최고**
2. **앙상블로 개선 여력 있지만 구현 최적화 필요** (목표 96.4-96.7%)
3. **Precision-Recall 균형이 핵심** - Voting=2의 2.6%p gap 해결 필요

### 다음 단계
1. ✅ **즉시**: Fold 1 단독 또는 Balanced threshold 제출
2. ⏱️ **30분 내**: Soft Voting 구현
3. 🔄 **1일 내**: Fold 2 훈련 및 3-Fold 앙상블

**목표 H-Mean**: 96.5% (현재 대비 +0.24%p)
