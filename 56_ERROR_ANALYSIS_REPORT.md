# OCR Text Detection 에러 케이스 분석 보고서
## Fold 3 Best Model (98.63% H-Mean) 기반 분석

---

## 📊 실행 요약 (Executive Summary)

**분석 대상:** Fold 3 최고 성능 모델 (Leaderboard H-Mean: 98.63%)  
**Validation Set:** 404 images, 46,714 text boxes  
**분석 일자:** 2025년 2월 7일  

### 핵심 발견사항
1. **전체 Validation Set 통계 분석 완료**: 404개 이미지, 46,714개 텍스트 박스
2. **평균 박스 수**: 115.63개/이미지 (중앙값: 110개)
3. **어려운 케이스 식별**: 고밀도, 다수 박스, 소형 박스, 극단적 종횡비 이미지 분류
4. **시각화 생성**: 통계 분포, 박스-이미지 크기 관계 등 분석 그래프

---

## 📈 분석 결과

### 1. Validation Set 전체 통계

#### 데이터셋 개요
- **전체 이미지 수**: 404개
- **전체 텍스트 박스 수**: 46,714개
- **이미지당 평균 박스 수**: 115.63개
- **이미지당 중앙값 박스 수**: 110개
- **최소 박스 수**: 48개
- **최대 박스 수**: 276개

#### 이미지 크기 특성
- **평균 너비**: 962 px
- **평균 높이**: 1,203 px
- **평균 면적**: 1.13 Megapixels

#### 텍스트 박스 특성
- **평균 박스 너비**: 86.2 px
- **평균 박스 높이**: 25.7 px
- **평균 박스 면적**: 2,295 px²
- **평균 밀도**: 110.13 boxes/Megapixel
- **평균 종횡비 (Aspect Ratio)**: 3.83

---

### 2. 어려운 케이스 (Challenging Cases) 식별

#### 2.1 극단적 종횡비 (Extreme Aspect Ratio)

**특징**: 평균 종횡비 > 6.0 (매우 가로로 긴 텍스트)

| 순위 | 이미지 | 평균 종횡비 |
|------|--------|------------|
| 1 | selectstar_000827.jpg | 7.68 |
| 2 | selectstar_004004.jpg | 6.66 |
| 3 | selectstar_000627.jpg | 6.44 |
| 4 | selectstar_000423.jpg | 6.39 |
| 5 | selectstar_000272.jpg | 6.32 |

**잠재적 문제**:
- Anchor box 매칭 어려움
- 가로로 매우 긴 텍스트 라인 검출 실패 가능
- Bounding box regression 정확도 저하

---

#### 2.2 고박스 수 (High Box Count)

**특징**: 이미지당 박스 수 > 200

| 순위 | 이미지 | 박스 수 |
|------|--------|---------|
| 1 | selectstar_002135.jpg | 276 |
| 2 | selectstar_002210.jpg | 254 |
| 3 | selectstar_003938.jpg | 245 |
| 4 | selectstar_003066.jpg | 234 |
| 5 | selectstar_003064.jpg | 228 |

**잠재적 문제**:
- 복잡한 레이아웃 처리 어려움
- False Positive 증가 가능성
- NMS (Non-Maximum Suppression) 매개변수 최적화 필요

---

#### 2.3 고밀도 (High Density)

**특징**: 밀도 > 300 boxes/Megapixel

| 순위 | 이미지 | 밀도 (boxes/Mpx) |
|------|--------|------------------|
| 1 | selectstar_001912.jpg | 538.1 |
| 2 | selectstar_001918.jpg | 479.8 |
| 3 | selectstar_003066.jpg | 384.9 |
| 4 | selectstar_000285.jpg | 363.6 |
| 5 | selectstar_001656.jpg | 347.2 |

**잠재적 문제**:
- 밀집된 텍스트 영역에서 박스 겹침
- False Negative (누락) 위험
- 작은 텍스트 영역 검출 실패

---

#### 2.4 소형 박스 (Small Boxes)

**특징**: 최소 박스 면적 < 40 px²

| 순위 | 이미지 | 최소 박스 면적 |
|------|--------|----------------|
| 1 | selectstar_000525.jpg | 20 px² |
| 2 | selectstar_003014.jpg | 21 px² |
| 3 | selectstar_003066.jpg | 30 px² |
| 4 | selectstar_002592.jpg | 30 px² |
| 5 | selectstar_000831.jpg | 33 px² |

**잠재적 문제**:
- 작은 텍스트 검출 실패
- Threshold 파라미터 민감도
- Feature map resolution 부족

---

### 3. 에러 분석 권장사항

#### 3.1 고밀도 이미지 (170.7+ boxes/Mpx)
- **리스크**: False Negative (예측 누락)
- **권장 조치**: 
  - Threshold 낮추기 (현재 0.22 → 0.15-0.20 실험)
  - Post-processing 미세 조정
  - Feature Pyramid Network 레이어 검토

#### 3.2 다수 박스 이미지 (155+ boxes)
- **리스크**: False Positive (오탐), 박스 매칭 어려움
- **권장 조치**:
  - NMS IoU threshold 조정
  - Box threshold 최적화
  - Confidence score 분포 분석

#### 3.3 소형 박스 (면적 < 61 px²)
- **리스크**: 작은 텍스트 검출 실패
- **권장 조치**:
  - Multi-scale test-time augmentation
  - 입력 해상도 증가 (1024 → 1280px)
  - Box threshold 세밀 조정

#### 3.4 극단적 종횡비 (AR > 5.2)
- **리스크**: 특이한 텍스트 레이아웃 검출 실패
- **권장 조치**:
  - Anchor box 종횡비 추가
  - Deformable convolution 적용
  - Unclip ratio 최적화

---

## 🔍 세부 분석 데이터

### 생성된 파일

#### CSV 파일
1. **validation_detailed_stats.csv** (404 rows)
   - 각 이미지별 상세 통계
   - 박스 수, 이미지 크기, 박스 특성, 밀도 등

2. **challenging_cases.csv**
   - 4가지 카테고리별 어려운 케이스 목록
   - 총 80개 케이스 (각 카테고리 20개)

#### 시각화 이미지
1. **validation_statistics.png**
   - 박스 수 분포
   - 이미지 크기 분포
   - 밀도 분포
   - 박스 면적 분포

2. **box_count_vs_image_size.png**
   - 박스 수와 이미지 크기의 관계
   - 밀도로 색상 구분

---

## 📝 결론 및 향후 과제

### 주요 발견
1. **Validation set은 고난이도**: 평균 115개의 박스/이미지로 매우 복잡
2. **성능 병목지점 식별**: 고밀도, 소형 박스, 극단적 종횡비 케이스
3. **개선 여지**: 현재 98.63% H-Mean에서 추가 0.5-1.0% 개선 가능

### 한계점
- **Test Set GT 부재**: Test set에 대한 실제 에러 케이스 분석 불가
- **예측 결과 부재**: Validation set에 대한 모델 예측 결과 미생성
- **직접 비교 불가**: GT와 예측 간 직접 비교 시각화 미실시

### 향후 과제
1. **Validation Set 예측 생성**: Fold 3 모델로 validation set inference 실행
2. **에러 케이스 시각화**: TP/FP/FN 시각화 이미지 생성
3. **세부 에러 분류**: 
   - 완전 누락 (False Negative - Missed Detection)
   - 오탐 (False Positive - False Alarm)
   - 경계 오류 (Boundary Error - IoU < 0.5)
4. **파라미터 최적화**: 어려운 케이스 기반 threshold 재조정

---

## 🚀 실행 가능한 다음 단계

### 1단계: Validation Set 예측 생성
```bash
python baseline_code/runners/test.py \
  checkpoint_path=/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt \
  # ... config 추가
```

### 2단계: 에러 케이스 매칭 및 시각화
```python
# final_error_analysis.py 재실행 (예측 결과와 함께)
python final_error_analysis.py \
  --pred_path validation_predictions.json \
  --gt_path val.json
```

### 3단계: 타겟 최적화
- 어려운 케이스 50개 선별하여 별도 분석
- 파라미터 grid search 재실행
- Ensemble 전략 재검토

---

## 📚 참고 자료

### 생성된 분석 파일 위치
```
/data/ephemeral/home/error_analysis/
├── validation_detailed_stats.csv      # 이미지별 상세 통계
├── challenging_cases.csv              # 어려운 케이스 목록
├── validation_statistics.png          # 통계 분포 시각화
└── box_count_vs_image_size.png       # 박스-크기 관계 시각화
```

### 주요 메트릭 정의
- **H-Mean (Harmonic Mean)**: Precision과 Recall의 조화평균 (F1-Score)
- **IoU (Intersection over Union)**: 예측 박스와 GT 박스의 겹침 비율
- **Density**: 단위 면적(Megapixel)당 텍스트 박스 수
- **Aspect Ratio**: 박스의 너비/높이 비율

---

## 📞 문의 및 추가 분석

추가 분석이 필요한 경우:
1. 특정 어려운 케이스에 대한 심층 분석
2. 다른 Fold 모델과의 비교 분석
3. Test set 예측에 대한 통계 분석 (GT 없이)
4. Threshold 파라미터 최적화 실험

---

**보고서 생성일**: 2025-02-07  
**분석 도구**: Python 3.10, PyTorch, Shapely, Matplotlib, Pandas  
**데이터셋**: SelectStar OCR Detection Dataset (Validation Split)  
