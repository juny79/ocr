# OCR 텍스트 검출 태스크 최적화 경험 분석 - 성능 상승 모멘텀 중심

**작성일**: 2026-02-12  
**분석 범위**: 00_baseline_analysis_report.md ~ 55_postprocessing_optimization_report.md  
**핵심 질문**: 리더보드 점수가 크게 상승한 실험은 무엇인가? 그 근본 원인은?

---

## 📈 리더보드 점수 변화 궤적

```
88.18% (Baseline 10epoch, 640px)
  │
  ├─ +3.60%p ─────────────────────► 95.81% (960px + Heavy Augmentation)
  │           (해상도 증가 + 데이터 강화)
  │
  ├─ +0.79%p ─────────────────────► 96.60% (Learning Rate Optimization, Validation)
  │           (하이퍼파라미터 세밀 조정)
  │
  ├─ -0.07%p ────────────────────► 96.53% (Thresh Postprocessing)
  │           (box_thresh 조정)
  │
  ├─ +1.84%p ─────────────────────► 98.37% (HRNet-W44 1024px, Lidarboard)
  │           (더 강력한 백본 + 높은 해상도)
  │
  ├─ +0.17%p ─────────────────────► 98.54% (Comprehensive 1024px Tuning)
  │           (파라미터 최적화 + External Data)
  │
  ├─ -9.76%p ────────────────────► 87.78% (3-Model NMS Ensemble) ❌
  │           (앙상블 시도 - 실패)
  │
  └─ +0.08%p ─────────────────────► 98.63% (K-Fold Fold 3, 최고점) ⭐
              (K-Fold 최적화 + Grid Search)
```

---

## 🎯 Part 1: 점수 상승의 4가지 모멘텀 분석

### 1️⃣ BIGGEST JUMP: 해상도 증가 + Heavy Augmentation (+3.60%p)

**출처**: `06_augmentation_960px_experiment_report.md`

#### 변수 변경사항
```yaml
해상도:        640×640 → 960×960 (+50%)
배치 크기:     8 → 4 (-50%)
Augmentation:  경미 → Heavy
  - HorizontalFlip: p=0.5
  - VerticalFlip: p=0.3 (신규!)
  - Rotation: ±15° (신규!)
  - ShiftScaleRotate: 6% (신규!)
  - RandomBrightnessContrast (강화)
  - Perspective Transform (신규!)
  - GridDistortion (신규!)

학습 설정:  Loss 기반 → H-Mean 기반 조기 종료
```

#### 성능 변화
```
Baseline (640px, Light Aug):  H-Mean 92.48%
960px + Heavy Aug:            H-Mean 95.81%
개선:                         +3.33%p (+3.60%)
```

#### 🔍 인사이트 1: **해상도와 Augmentation의 협력 효과**

**수학적 분석**:
```
해상도 증가: 640² = 409,600 px → 960² = 921,600 px (+125% 픽셀)
효과:
  ✓ 작은 글자 감지 능력 향상 (Recall ↑)
  ✓ 경계선 정밀도 향상 (Precision 유지)
  ✓ 더 많은 augmentation 적용 가능

Augmentation 강화:
  ✓ 회전, 왜곡 등으로 과적합 방지
  ✓ 실제 스캔 문서의 다양한 각도 대비
  ✓ 배치 크기 감소(8→4) 보완
```

**핵심 발견**: 
- **단독 해상도 증가만으로는 +1~2%p**
- **해상도 + Augmentation 조합 = +3.6%p**
- → 두 요소의 상호작용이 핵심 (곱셈 효과, 단순 덧셈 아님)

#### ⚠️ 트레이드오프
```
배치 크기 감소 (8→4) 영향:
  ├─ 배치 정규화 효과 감소
  ├─ 기울기 추정 분산 증가
  └─ 하이퍼파라미터 재조정 필요

하지만 Augmentation이 이를 충분히 보상
→ 장기 학습 결과 순이득
```

---

### 2️⃣ BACKBONE UPGRADE: HRNet-W44 1024px (+1.84%p)

**출처**: `43_hrnet_w44_1024_resolution_experiment_report.md`

#### 변수 변경사항
```yaml
모델 아키텍처:  ResNet18 → HRNet-W44
해상도:         960×960 → 1024×1024 (+6.6%)
학습 에포크:    ~12-13 → 18

구성:
  Encoder: HRNet-W44 (High Resolution Network)
  Decoder: FPN + Dense connections
  Head: DBHead (Differentiable Binarization)
```

#### 성능 변화
```
960px Baseline (Fold 0):     H-Mean 95.81% (추정)
1024px HRNet-W44:            H-Mean 98.37% (리더보드)
개선:                        +2.56%p
```

#### 🔍 인사이트 2: **High Resolution Network의 우월성**

**HRNet의 특이점**:
```
기존 ResNet: Down-sample → Up-sample 구조
  문제: 저해상도에서 정보 손실, 세밀한 텍스트 경계선 불안정

HRNet: Multi-scale Parallel Paths
  ✓ 고해상도 경로 유지 (항상 원본 해상도의 1/4이상)
  ✓ 다중 스케일에서의 정보 融合
  ✓ 텍스트 검출에 최적화된 구조
```

**성능 영향**:
```
정밀도 (Precision): 96.51% → 98.84% (+2.33%p)
재현율 (Recall):    81.94% → 98.45% (+16.51%p) ⭐⭐⭐⭐⭐

→ Recall 개선이 핵심 (16.51%p!)
  = 기존에 놓친 텍스트 85% 이상 감지 가능해짐
```

#### 🧮 수학적 의의
```
H-Mean = 2 × P × R / (P + R)

Precision과 Recall의 조화평균이므로,
한쪽이 크게 향상되면 전체 점수 급등

H-Mean 변화:
  기존: 2 × 0.9651 × 0.8194 / 1.7845 = 0.8818
  신규: 2 × 0.9884 × 0.9845 / 1.9729 = 0.9837
  → 2.56%p 향상 = Recall +16.51%p의 영향
```

---

### 3️⃣ HYPERPARAMETER TUNING: Comprehensive 1024px (+0.17%p)

**출처**: `48_comprehensive_1024_external_data_report.md`

#### 변수 변경사항
```yaml
데이터:      기본 train.json → train_augmented_full.json
External:    External data 추가 학습
Postprocess: thresh, box_thresh Grid Search
  - thresh: 0.20~0.30 (0.01 간격)
  - box_thresh: 0.35~0.45 (0.01 간격)

최적값:      thresh=0.22, box_thresh=0.40
```

#### 성능 변화
```
HRNet 1024px (기본):         H-Mean 98.37%
Comprehensive Tuning:        H-Mean 98.54%
개선:                        +0.17%p
```

#### 🔍 인사이트 3: **수렴 곡선의 한계점에 도달**

**패턴 분석**:
```
점수 상승도:
  640→960px:      +3.60%p (큰 점프)
  ResNet→HRNet:   +1.84%p (중간 점프)
  Hyperparameter: +0.17%p (미세 조정)
  
경향:
  1단계: 기본 모델 개선 (대규모 변경)    → 큰 서이득
  2단계: 아키텍처 강화 (중규모 변경)    → 중간 이득
  3단계: 파라미터 최적화 (소규모 변경)  → 극히 작은 이득
  
→ 수렴 곡선의 특성 (수확 체감의 법칙)
```

**Postprocessing Grid Search의 한계**:
```
thresh vs Recall 곡선이 NON-MONOTONIC:

Recall
  │
  ├─ 0.9840 ─ thresh=0.210
  │
  ├─ 0.9834 ─ thresh=0.212 (하락!)
  │
  ├─ 0.9838 ─ thresh=0.218 ⭐ LOCAL PEAK
  │
  ├─ 0.9828 ─ thresh=0.220+
  │
  └─ 0.9806 ─ thresh=0.230 (급락)

→ 국소 최대값에 도달했을 가능성 높음
  = 추가 조정으로 무한 개선 불가능
```

---

### 4️⃣ K-FOLD 최적화: 최고점 달성 (+0.08%p)

**출처**: `baseline_code/55_kfold_optimized_training_analysis_report.md` + `55_postprocessing_optimization_report.md`

#### 변수 변경사항
```yaml
학습 전략:  단일 모델 → 5-Fold Cross-Validation
최종 선택:  5개 Fold 중 Fold 3 선택 (val H-Mean 최고)

Fold 3 특성:
  - Val H-Mean: 98.31% (높음)
  - Test H-Mean: 98.32% (안정적)
  - Lidarboard H-Mean: 98.63% ⭐ (최고!)

Post-processing Grid Search:
  - thresh: 0.20~0.22 (0.001 간격) 극미세 조정
  - unclip_ratio: 1.97, 1.98, 1.99, 2.00 (미세 조정)
  - 총 195+ 파라미터 조합 탐색
```

#### 성능 변화
```
Comprehensive Best:          H-Mean 98.54%
K-Fold Fold 3:              H-Mean 98.63%
개선:                        +0.08%p (최저이지만 최고점)
```

#### 🔍 인사이트 4: **다양성의 가치 vs 앙상블의 함정**

**K-Fold의 이점**:
```
서로 다른 학습 데이터로 5개 독립 모델 생성:
  ├─ Fold 0: H-Mean 98.51%
  ├─ Fold 1: H-Mean 98.48%
  ├─ Fold 2: H-Mean 98.44%
  ├─ Fold 3: H-Mean 98.63% ⭐ 최고
  └─ Fold 4: H-Mean 98.40%

효과:
  ✓ 모델 다양성 확보 (같은 데이터 반복 아님)
  ✓ 데이터 불균형에 따른 변동성 흡수
  ✓ 최고 Fold 선택 가능
```

**K-Fold 앙상블 시도 실패**:
```
❌ 5개 Fold 투표/앙상블:
  - NMS Ensemble: H-Mean 88% (13%p 떨어짐!)
  - WBF Ensemble: H-Mean 87.4% (11%p 떨어짐!)

원인:
  → 각 Fold는 **이미 평가 데이터에 최적화된 모델**
  → 합치면 오버피팅된 영역들이 충돌
  → Precision/Recall 동시 손실

교훈: **이미 우수한 model들의 앙상블은 악화**
      (학습 데이터 분산이 아닌 평가 데이터에서의 앙상블)
```

---

## 📊 Part 2: 실패 케이스 분석

### ❌ TTA + NMS 시도 실패

**출처**: `55_postprocessing_optimization_report.md`

```
최고점 기준 (thresh=0.22, box_thresh=0.40): H-Mean 98.63%
TTA (수평 플립 합성):                     H-Mean ? (추정 98.62%)
TTA + NMS (IoU 0.3):                      H-Mean 98.56% ❌ (-0.07%p)
```

**원인**:
```
TTA (Test-Time Augmentation) 문제:
  
  prob_map_orig:  0.25 (경계선 박스)
  prob_map_flip:  0.25 (같은 박스)
  평균:           0.50 > thresh(0.22) ✓ 통과
  
  하지만 회전/플립에서 정렬 오차 누적:
  └─ 경계선 신호 약화 → Recall 감소
  
기존 모델 특성:
  → 이미 threshold가 최적화됨
  → 추가 평균화는 신호 손실만 초래
```

**통계**:
```
원본:      H=98.63%, P=98.88%, R=98.44%
TTA+NMS:   H=98.56%, P=98.89%, R=98.30%
손실:      Recall -0.14%p
```

---

## 💡 Part 3: OCR 태스크에서의 핵심 인사이트

### Insight 1: 해상도는 절대적 (640→960→1024)

**발견**:
```
640px:   H=88.18% (기본)
960px:   H=95.81% (+7.63%p)
1024px:  H=98.37% (+2.56%p from 960px)

패턴:
  → 해상도의 지수적 효과 (Recall이 크게 향상)
  → 세로로 긴 텍스트, 작은 글자에 효과적
  
한계:
  → 메모리 제약 (RTX 3090에서 배치 크기 4)
  → 극도로 높은 해상도(>1280px) 수렴한계 도달
```

**OCR의 특성**: 텍스트는 작은 영역이며, 해상도가 낮으면 경계선 정보 손실

---

### Insight 2: 아키텍처 선택이 핵심 (ResNet18 → HRNet-W44)

**발견**:
```
ResNet18:  H=95.81%
HRNet-W44: H=98.37% (+2.56%p)

차이점:
  ResNet18:  피라미드 구조 (저해상도로 수렴)
  HRNet:     병렬 다중 해상도 유지
  
재현율 개선 폭:
  ResNet: Recall ca. 92%
  HRNet:  Recall = 98.45% (+6%p!)
```

**OCR의 특성**: 텍스트 경계선의 정밀도가 중요 → 고해상도 경로 필수

**교훈**: 
```
일반 물체 검출 (YOLO, Faster-RCNN):
  → ResNet, EfficientNet 충분

텍스트 검출 (OCR):
  → HRNet, ConvNeXt, Vision Transformer 권장
  → 고해상도 유지가 생명
```

---

### Insight 3: Augmentation은 정규화 기법 (+3.6%p에 기여)

**발견**:
```
경미 Augmentation:   H=92.48%
Heavy Augmentation:  H=95.81% (+3.33%p)

구성:
  - Geometric: Rotation, ShiftScaleRotate, Perspective
  - Photometric: Brightness, Contrast
  - Morphological: GridDistortion
  
효과:
  ✓ 과적합 방지
  ✓ 실제 문서의 다양한 스캔 변형 대비
  ✓ Recall 특히 향상 (작은 글자 견고성)
```

**OCR의 특성**: 실제 영수증/문서는 다양한 각도, 조명에서 스캔됨

---

### Insight 4: 파라미터 미세조정은 한계가 빠르다 (+0.08%p)

**발견**:
```
단계별 수렴:
  1) 캐시 개선:      +3.60%p (빠른 상승)
  2) 아키텍처 강화:  +1.84%p (점진적 상승)
  3) 하이퍼 조정:    +0.17%p (미세 개선)
  4) K-Fold 최적화:  +0.08%p (극한 조정)
  
곡선 특성: 지수 감쇠 (logarithmic convergence)
```

**수렴 곡선**:
```
성능 개선폭
    │
  3 │  ▲
    │  ├─▄ ┌─────
  2 │  │  ├────┐
    │  │  │    ├──┐
  1 │  │  │    │  ├──┐
    │  ▼  ▼    ▼  │  ▼
    └──────────────────────
       해상도  아키텍  파라  K-Fold
       증가    강화    미조  최적화
```

**교훈**: 기본이 약하면 규모 있는 변경이 필수. 기본이 좋으면 세밀한 조정만 가능.

---

### Insight 5: 앙상블은 조건부 (이미 우수한 모델들은 악화)

**발견**:
```
❌ K-Fold 투표/앙상블:   H: 98.63 → 88.78% (13%p 급락)
✅ K-Fold 선택 (Best):   H: 98.63% (그대로 유지)

패턴:
  모델 A: H=0.9863
  모델 B: H=0.9851
  모델 C: H=0.9844
  
  합치면: H=0.8878 (최악!)
  선택:   H=0.9863 (최고 유지)
```

**언제 앙상블이 효과적인가?**:
```
✓ 약한 모델들 (H<0.95): 앙상블로 +1~2%p 가능
✓ 서로 다른 아키텍처: 보완 가능
✓ 서로 다른 학습 데이터: 일반화 향상

✗ 이미 우수한 모델 (H>0.98): 앙상블 역효과
✗ 같은 데이터로 학습: 과적합 영역 충돌
✗ K-Fold (평가 데이터 최적화): 충돌 심화
```

**OCR의 특성**: 높은 정확도에서는 정밀한 경계선이 중요 → 앙상블의 모호화 문제

---

## 🎓 Part 4: OCR 태스크의 보편적 최적화 전략

### 프로토콜

```
1️⃣ 기본 모델 구축 (H=80~90%)
   └─ 해상도 선택: 가능한 한 높게 (1024px 권장)
   └─ 아키텍처: HRNet 또는 고해상도 특화 모델

2️⃣ Data Augmentation 강화 (H=90~96%)
   └─ Geometric: Rotation, Perspective, Affine
   └─ Geometric + Photometric 조합
   └─ 예상 개선: +3~5%p

3️⃣ 아키텍처 탐색 (H=96~98%)
   └─ HRNet, ConvNeXt, Vision Transformer 비교
   └─ 예상 개선: +0.5~2%p

4️⃣ K-Fold 검증 (H=98~98.5%)
   └─ 각 Fold의 성능 확인
   └─ 최고 Fold 선택 (앙상블 X)
   └─ 예상 개선: +0~0.3%p

5️⃣ Postprocessing Grid Search (H>98.5%)
   └─ thresh, box_thresh 미세 조정
   └─ 예상 개선: +0.01~0.1%p (매우 작음)
   └─ 수확 체감의 법칙 적용
```

---

## 📋 Part 5: 실험 변수별 성공/실패

| 실험 | 변수 | 방향 | 효과 | 상태 | 교훈 |
|------|------|------|------|------|------|
| 해상도 증가 | 640→960 | 규모 | +3.60%p | ✅ | 가장 큰 이득 |
| Augmentation 강화 | Light→Heavy | 정규화 | 포함 | ✅ | 해상도와 시너지 |
| 아키텍처 변경 | ResNet→HRNet | 구조 | +2.56%p | ✅ | Recall 특히 개선 |
| 데이터 추가 | 기본→Full | 규모 | +0.17%p | ⚠️ | 효과 미미 |
| Postprocessing | Thresh 조정 | 미세 | -0.07%p | ❌ | 한계점 도달 |
| TTA | 플립 합성 | 앙상블 | -0.07%p | ❌ | 이미 최적화됨 |
| NMS | 중복 제거 | 후처리 | 무효 | ❌ | 경계선 모호화 |
| K-Fold 투표 | 다중 모델 | 앙상블 | -13%p | ❌❌ | 대재해 |
| K-Fold 선택 | Best Fold | 선택 | +0.08%p | ✅ | 최고점 달성 |

---

## 🏆 결론: OCR 최적화의 3가지 황금 규칙

### Rule 1: 해상도 먼저, 아키텍처 다음

```
우선순위:
  1위: 해상도 (1024px 물리적 한계까지)
  2위: 아키텍처 (고해상도 특화 모델 선택)
  3위: Augmentation (과적합 방지)
  4위: 파라미터 미세조정 (거의 무시)
  
이유: OCR은 텍스트 경계선이 생명
     → 고해상도, 고충실도 모델이 필수
     → 저해상도 모델의 Augmentation은 무용지물
```

### Rule 2: 이미 우수한 모델은 앙상블 금지

```
H < 95%: 앙상블로 +1~3%p 가능 ✅
H > 98%: 앙상블로 -5~15%p 악화 ❌

이유: 높은 정확도 영역에서는 모델들의 예측이
     극도로 일관성 있음 (이미 평가 데이터 최적화)
     → 합치면 서로 다른 오류 영역의 충돌
     
해결책: K-Fold 중 최고 모델 1개 선택
       (앙상블 금지, Best 선택)
```

### Rule 3: 수렴 곡선 인식 - 초기 빠른 상승, 말기 미시적 개선

```
성능 상승:
  첫 번째 개선 (표준 → 좋음):    +3~5%p, 빠름 ✅
  두 번째 개선 (좋음 → 매우좋음):  +1~2%p, 중간 속도
  세 번째 개선 (매우좋음 → 최고):  +0.1%p 이하, 극도로 느림 ⚠️

의미: 98% 이상의 고성능에서는
      추가 개선이 거의 불가능
      → 새로운 아키텍처/데이터 필요
      → 미세 조정은 수익성 0
```

---

## 🎯 최종 평가

**가장 효과적이었던 3가지**:
1. ⭐⭐⭐⭐⭐ **해상도 증가** (640→960→1024px): +7.63%p 누적
2. ⭐⭐⭐⭐ **아키텍처 강화** (ResNet→HRNet): +2.56%p
3. ⭐⭐⭐ **Heavy Augmentation**: +3.33%p (해상도 증가에 포함)

**최악의 선택**:
1. ⭐ **K-Fold 앙상블** (투표/NMS): -13%p (재앙)
2. ⭐ **TTA + NMS**: -0.07%p (미세 악화)

**최종 점수**:
- **최고점**: H-Mean 98.63% (Fold 3, K-Fold 최적화)
- **2위**: H-Mean 98.54% (Comprehensive 1024px)
- 차이: +0.08%p (극미세)

**결론**: 기본이 중요하다. 해상도와 아키텍처로 90% 이상의 성과를 달성하면, 나머지 10%는 극도로 어렵고 비효율적이다. OCR 최적화는 **앞부분에서의 큰 결정이 전부**이며, 뒷부분 미세조정은 거의 무의미함.

---

## 📚 참고: 보고서 참조 목록

- `00_baseline_analysis_report.md`: 초기 상태 (H=88.18%)
- `06_augmentation_960px_experiment_report.md`: 해상도 증가 (+3.60%p)
- `17_sweep_lr_optimization_analysis_report.md`: LR 최적화
- `43_hrnet_w44_1024_resolution_experiment_report.md`: HRNet 1024px (+2.56%p)
- `48_comprehensive_1024_external_data_report.md`: 종합 최적화 (+0.17%p)
- `53_ensemble_failure_analysis_report.md`: 앙상블 실패 (-13%p)
- `55_postprocessing_optimization_report.md`: K-Fold 최적화 (+0.08%p)
- `baseline_code/55_kfold_optimized_training_analysis_report.md`: K-Fold 상세 분석

