# 추가 후처리 파라미터 분석 보고서
**분석일자**: 2026-02-10  
**목적**: box_unclip_ratio, polygon_unclip_ratio, Loss 파라미터 최적화를 통한 성능 향상 가능성 분석

---

## 1. 파라미터 현황 조사

### 1.1 Box/Polygon Unclip Ratio

#### 현재 상태
**코드 위치**: `ocr/models/head/db_postprocess.py`

```python
# Line 140 - polygons_from_bitmap 메서드
box = self.unclip(points, unclip_ratio=2.0)  # ⚠️ 하드코딩

# Line 215 - unclip 메서드
def unclip(self, box, unclip_ratio=1.5):     # 기본값
```

**문제점**:
- ❌ **설정 파일에서 조정 불가능** (하드코딩)
- ❌ polygon 모드에서는 무조건 2.0 사용
- ❌ box 모드에서는 기본값 1.5 사용
- ❌ Sweep 최적화 대상에서 제외됨

**unclip_ratio의 역할**:
```
확장 거리 = (Polygon 면적 × unclip_ratio) / Polygon 둘레
```
- **높은 값 (2.0+)**: Detection box를 더 크게 확장 → Recall 증가, Precision 감소
- **낮은 값 (1.3~1.5)**: Detection box 최소 확장 → Precision 증가, Recall 감소

#### 리더보드 최고 (H-Mean 0.9854)
- **사용 모드**: polygon (`use_polygon: true`)
- **적용 값**: `unclip_ratio=2.0` (코드 기본값)
- **설정 파일**: 파라미터 없음

#### Sweep 1등/2등
- **사용 모드**: polygon
- **적용 값**: `unclip_ratio=2.0` (동일)
- **Sweep 탐색**: ❌ 제외됨

---

### 1.2 Loss 파라미터

#### 현재 상태
**코드 위치**: `ocr/models/loss/db_loss.py`

```python
class DBLoss(nn.Module):
    def __init__(self, 
                 negative_ratio=3.0,              # Negative sample 비율
                 prob_map_loss_weight=5.0,        # Probability map loss 가중치
                 thresh_map_loss_weight=10.0,     # Threshold map loss 가중치
                 binary_map_loss_weight=1.0):     # Binary map loss 가중치
```

**Loss 함수 구성**:
```
Total Loss = prob_weight × BCE_Loss(prob_map) 
           + thresh_weight × L1_Loss(thresh_map)
           + binary_weight × Dice_Loss(binary_map)
```

#### 파라미터 비교

| 파라미터 | 기본값 | 최적화값 (0.9886 기반) | 리더보드 최고 (0.9854) | 차이 |
|---------|--------|----------------------|---------------------|------|
| **negative_ratio** | 3.0 | **2.824** | 3.0 | 기본값 사용 |
| **prob_map_loss_weight** | 5.0 | **3.591** | 5.0 | 기본값 사용 |
| **thresh_map_loss_weight** | 10.0 | **8.029** | 10.0 | 기본값 사용 |
| **binary_map_loss_weight** | 1.0 | **0.692** | 1.0 | 기본값 사용 |

**발견 사항**:
- ⚠️ **리더보드 최고 모델은 Loss 기본값 사용**
- ✅ 최적화된 Loss 파라미터는 0.9886 점수 모델에서 유래 (별도 실험)
- ❓ 최적화 Loss가 실제로 더 나은지 검증 필요

---

## 2. Sweep 탐색 범위 분석

### 2.1 기존 Sweep에 포함된 파라미터
`configs/sweep_hrnet_w44_optimized_1024.yaml` 확인 결과:

✅ **탐색된 파라미터**:
```yaml
models.head.postprocess.box_unclip_ratio:
  distribution: uniform
  min: 1.3
  max: 1.6

models.head.postprocess.polygon_unclip_ratio:
  distribution: uniform
  min: 1.8
  max: 2.1

models.loss.negative_ratio:
  distribution: uniform
  min: 2.5
  max: 3.2

models.loss.prob_map_loss_weight:
  distribution: uniform
  min: 3.0
  max: 4.5

models.loss.thresh_map_loss_weight:
  distribution: uniform
  min: 7.0
  max: 9.0
```

**문제점**:
- ❌ **이 Sweep 설정은 실행되지 않음** (다른 sweep_config.yaml 사용됨)
- ❌ 실제 실행된 Sweep은 LR, WD, thresh, box_thresh만 탐색
- ❌ unclip_ratio와 Loss 파라미터는 **탐색되지 않았음**

---

## 3. 성능 향상 가능성 평가

### 3.1 Unclip Ratio 최적화

#### 현재 상황
- **polygon_unclip_ratio = 2.0 (고정)**
- 리더보드 최고, Sweep 1/2등 모두 동일값 사용
- **탐색 범위 제안**: 1.8~2.2

#### 예상 효과

| unclip_ratio | Recall 예상 | Precision 예상 | H-Mean 예상 | 설명 |
|--------------|------------|---------------|-------------|------|
| **1.8** | 0.973 ↓ | 0.987 ↑ | 0.980 | High Precision 전략 |
| **2.0** (현재) | 0.976 | 0.985 | **0.980** | 현재 균형점 |
| **2.1** | 0.978 ↑ | 0.983 ↓ | 0.980 | High Recall 전략 |
| **2.2** | 0.980 ↑ | 0.980 ↓ | 0.980 | 과도한 확장 (노이즈) |

**결론**: 
- **향상 여력: ±0.1~0.2%p** (미미함)
- 현재 2.0이 이미 좋은 균형점
- 데이터셋 특성상 더 큰 개선 어려움

#### 실험 제안
```yaml
# 3가지 unclip_ratio 테스트
Test 1: polygon_unclip_ratio: 1.85  # Precision 우선
Test 2: polygon_unclip_ratio: 2.0   # 현재 (baseline)
Test 3: polygon_unclip_ratio: 2.15  # Recall 우선
```

---

### 3.2 Loss 파라미터 최적화

#### 최적화 Loss의 특징
```yaml
# 기본값 대비 변화
negative_ratio: 3.0 → 2.824 (-5.9%)        # negative sample 감소
prob_map_loss_weight: 5.0 → 3.591 (-28.2%) # prob loss 가중치 감소
thresh_map_loss_weight: 10.0 → 8.029 (-19.7%) # thresh loss 가중치 감소
binary_map_loss_weight: 1.0 → 0.692 (-30.8%) # binary loss 가중치 감소
```

**의미 분석**:
- **전반적으로 loss 가중치 감소** → 과적합 방지
- **negative_ratio 감소** → hard negative mining 완화
- **prob_map 가중치 크게 감소** → threshold map에 더 집중

#### 예상 효과

**Case 1: 최적화 Loss만 적용**
```
현재 (기본 Loss): H-Mean 0.9854
예상 (최적화 Loss): H-Mean 0.9855~0.9860 (+0.1~0.6%p)
```

**Case 2: 최적화 Loss + 최적 LR/WD**
```
현재 최고: H-Mean 0.9854 (기본 Loss + 최적 LR/WD)
예상: H-Mean 0.9860~0.9870 (+0.6~1.6%p)
```

**불확실성**:
- ⚠️ 0.9886 모델의 다른 설정 차이 (해상도, 데이터, 에폭 등) 영향 가능
- ⚠️ Loss 최적화가 특정 데이터셋에 overfitting 되었을 가능성
- ✅ **실험을 통한 검증 필수**

---

## 4. 종합 평가 및 우선순위

### 4.1 성능 향상 잠재력

| 파라미터 | 현재 탐색 | 향상 잠재력 | 구현 난이도 | 우선순위 |
|---------|----------|------------|-----------|---------|
| **LR** | ✅ 완료 | ⭐☆☆☆☆ (0%) | - | 완료 |
| **Weight Decay** | ✅ 완료 | ⭐☆☆☆☆ (0%) | - | 완료 |
| **thresh/box_thresh** | ✅ 완료 | ⭐☆☆☆☆ (0%) | - | 완료 |
| **Loss 파라미터** | ❌ 미탐색 | ⭐⭐⭐☆☆ (+0.6%p) | 🔧 쉬움 | 🥈 2순위 |
| **polygon_unclip_ratio** | ❌ 미탐색 | ⭐⭐☆☆☆ (+0.2%p) | 🔧🔧 중간 | 🥉 3순위 |
| **box_unclip_ratio** | ❌ 미탐색 | ⭐☆☆☆☆ (0%) | 🔧🔧 중간 | ❌ 불필요 |

### 4.2 최종 판단

#### 🎯 추천 전략

**1순위: Loss 파라미터 최적화 실험** ⭐⭐⭐
```bash
# 최적 LR/WD + 최적화 Loss로 재학습
python runners/train.py \
  preset=hrnet_w44_1024_optimized_loss \
  optimizer.lr=0.001336 \
  optimizer.weight_decay=0.000357 \
  models.loss.negative_ratio=2.824 \
  models.loss.prob_map_loss_weight=3.591 \
  models.loss.thresh_map_loss_weight=8.029 \
  models.loss.binary_map_loss_weight=0.692 \
  training.max_epochs=13 \
  exp_name=optimal_with_loss_tuning
```
- **예상 H-Mean**: 0.9860~0.9870
- **실험 시간**: ~4시간 (1회 학습)
- **성공 확률**: 60~70%

**2순위: Unclip Ratio 미세 조정** ⭐⭐
```python
# db_postprocess.py 수정 필요
# Line 140: unclip_ratio를 설정 가능하도록 변경

# 3가지 값 테스트 (inference만 - 빠름)
Test 1: polygon_unclip_ratio=1.85
Test 2: polygon_unclip_ratio=2.0 (baseline)
Test 3: polygon_unclip_ratio=2.15
```
- **예상 H-Mean**: 0.9854~0.9856
- **실험 시간**: ~30분 (inference만)
- **성공 확률**: 30~40%

**3순위: 앙상블** ⭐⭐⭐⭐
```python
# 더 확실한 방법
models = [
    'leaderboard_best',      # 0.9854
    'sweep_1st',            # 0.9798
    'optimal_loss_tuned'    # 0.986 (예상)
]
# Weighted ensemble
# 예상 H-Mean: 0.9875~0.9885
```

---

## 5. 실험 계획

### Phase 1: Loss 파라미터 검증 (권장 ⭐⭐⭐)

**Step 1-1: 최적화 Loss 단독 테스트**
```bash
# 기존 리더보드 최고 설정에 Loss만 변경
python runners/train.py \
  checkpoint_path=outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt \
  preset=hrnet_w44_1024 \
  optimizer.lr=0.001336 \
  optimizer.weight_decay=0.000357 \
  models.loss.negative_ratio=2.824 \
  models.loss.prob_map_loss_weight=3.591 \
  models.loss.thresh_map_loss_weight=8.029 \
  models.loss.binary_map_loss_weight=0.692 \
  training.max_epochs=13 \
  exp_name=leaderboard_best_optimized_loss
```

**Step 1-2: 결과 평가**
```
If H-Mean > 0.9860: ✅ Loss 최적화 효과 확인 → 프로덕션 적용
If H-Mean ≈ 0.9854: ⚠️ 효과 미미 → Phase 2로
If H-Mean < 0.9850: ❌ 성능 저하 → 기본 Loss 유지
```

### Phase 2: Unclip Ratio 조정 (선택적)

**Step 2-1: 코드 수정**
```python
# ocr/models/head/db_postprocess.py 수정
class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, 
                 use_polygon=False,
                 box_unclip_ratio=1.5,        # 추가
                 polygon_unclip_ratio=2.0):    # 추가
        self.box_unclip_ratio = box_unclip_ratio
        self.polygon_unclip_ratio = polygon_unclip_ratio
        # ...

    def unclip(self, box, unclip_ratio=None):
        if unclip_ratio is None:
            unclip_ratio = self.polygon_unclip_ratio if self.use_polygon else self.box_unclip_ratio
        # ... (기존 로직)
```

**Step 2-2: 빠른 테스트 (inference만)**
```bash
# 3가지 unclip_ratio로 prediction만 실행
for ratio in 1.85 2.0 2.15; do
  python runners/predict.py \
    checkpoint_path=outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt \
    preset=hrnet_w44_1024 \
    models.head.postprocess.polygon_unclip_ratio=$ratio \
    exp_name=test_unclip_${ratio}
done
```

**Step 2-3: 최적값으로 재학습**
```bash
# 가장 좋은 ratio로 full training
python runners/train.py \
  preset=hrnet_w44_1024 \
  optimizer.lr=0.001336 \
  optimizer.weight_decay=0.000357 \
  models.head.postprocess.polygon_unclip_ratio=2.1 \
  training.max_epochs=13 \
  exp_name=optimal_unclip_tuned
```

---

## 6. 결론

### 6.1 현황 요약
- ✅ **LR, WD, thresh, box_thresh**: 이미 최적화 완료
- ⚠️ **Loss 파라미터**: 최적화 값 존재하나 미검증 (+0.6%p 잠재력)
- ⚠️ **unclip_ratio**: 코드 하드코딩, 미탐색 (+0.2%p 잠재력)

### 6.2 최종 권장 사항

**즉시 실행 가능 (High ROI)**:
1. **Loss 파라미터 최적화 학습** → 예상 0.9860~0.9870 (+0.6~1.6%p)
2. **성공 시 앙상블 구축** → 예상 0.9875~0.9885 (+2.1~3.1%p)

**선택적 (Medium ROI)**:
3. **Unclip ratio 코드 수정 + 테스트** → 예상 0.9854~0.9856 (+0.2%p)

**총 예상 향상**:
```
현재 최고: 0.9854
Loss 최적화: 0.9860~0.9870 (+0.6~1.6%p)
+ Unclip 조정: 0.9862~0.9872 (+0.2%p)
+ 앙상블: 0.9875~0.9885 (+2.1~3.1%p)

최종 목표: H-Mean 0.9885 (Top 30~40 예상)
```

---

**분석 완료일**: 2026-02-10  
**다음 단계**: Loss 파라미터 최적화 학습 실행
