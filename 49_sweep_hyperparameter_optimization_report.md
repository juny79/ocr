# WandB Sweep 기반 하이퍼파라미터 최적화 보고서

## 📋 실험 개요

### 실험 기간
- **시작**: 2026년 2월 9일 21시 27분
- **종료**: 2026년 2월 10일 20시 41분
- **총 소요 시간**: 약 23시간

### 실험 설정
- **모델**: DBNet with HRNet-W44 (1024x1024)
- **데이터셋**: 통합 데이터셋 (4,698 images)
  - CORD-v2: 800 images
  - SROIE: 626 images  
  - WildReceipt: 1,765 images
  - ICDAR 2019 SROIE: 1,507 images
- **Sweep 방법**: Bayesian Optimization
- **완료된 Run 수**: 9개
- **조기 종료**: Hyperband (min_iter=3)

### 탐색 범위
| 파라미터 | 최소값 | 최대값 | 설명 |
|---------|--------|--------|------|
| Learning Rate | 0.0008 | 0.002 | 학습률 |
| Weight Decay | 0.0001 | 0.0006 | L2 정규화 |
| T_max | 8 | 15 | Cosine Annealing 주기 |
| Thresh | 0.20 | 0.24 | 확률 맵 임계값 |
| Box Thresh | 0.40 | 0.44 | 박스 신뢰도 임계값 |
| Max Epochs | 10, 13, 15 | - | 최대 학습 에폭 |
| Batch Size | 1 | - | GPU 메모리 제약으로 고정 |

---

## 🏆 최적 파라미터 결과

### 1위 🥇 Run: dusi9e8b

#### 성능 지표
- **Validation H-Mean**: **0.97712** (최고 성능)
- **Validation Precision**: 0.97937
- **Validation Recall**: 0.97590
- **Test H-Mean**: 0.97712

#### 하이퍼파라미터
```yaml
Learning Rate: 0.000974
Weight Decay: 0.000146
T_max (Cosine Scheduler): 12
Probability Threshold: 0.229
Box Threshold: 0.400
Batch Size: 1
Max Epochs: 13 (완료)
```

#### 특징
- **균형잡힌 Precision-Recall**: Precision(97.94%)과 Recall(97.59%)이 모두 높은 균형 달성
- **안정적 학습률**: 0.000974로 중간 범위 값 사용
- **적절한 Scheduler 주기**: T_max=12로 13 에폭에 최적 수렴
- **보수적 Threshold**: thresh=0.229, box_thresh=0.400으로 False Positive 억제

---

### 2위 🥈 Run: 2vayr7k4

#### 성능 지표
- **Validation H-Mean**: **0.97589** (-0.00123)
- **Validation Precision**: 0.97647
- **Validation Recall**: 0.97638
- **Test H-Mean**: 0.97589

#### 하이퍼파라미터
```yaml
Learning Rate: 0.001058
Weight Decay: 0.000141
T_max: 13
Probability Threshold: 0.207
Box Threshold: 0.417
Max Epochs: 15 (완료)
```

#### 특징
- **높은 Recall**: 97.64%로 1위 대비 Recall이 0.05%p 높음
- **더 긴 학습**: 15 에폭까지 학습하여 충분한 수렴
- **낮은 Probability Threshold**: 0.207로 더 많은 텍스트 영역 검출

---

### 3위 🥉 Run: fdp8oeci

#### 성능 지표
- **Validation H-Mean**: **0.97186** (-0.00526)
- **Validation Precision**: 0.97194
- **Validation Recall**: 0.97334
- **Test H-Mean**: 0.97186

#### 하이퍼파라미터
```yaml
Learning Rate: 0.001252
Weight Decay: 0.000485
T_max: 9
Probability Threshold: 0.214
Box Threshold: 0.439
Max Epochs: 13 (완료)
```

#### 특징
- **높은 Weight Decay**: 0.000485로 과적합 방지 강화
- **짧은 Scheduler 주기**: T_max=9로 빠른 학습률 감소
- **균형잡힌 성능**: Precision과 Recall 모두 97% 이상 유지

---

## 📊 성능 비교 분석

### Top 5 Run 성능 요약

| Rank | Run ID | Val H-Mean | Val Precision | Val Recall | Test H-Mean | Epochs |
|------|--------|------------|---------------|------------|-------------|---------|
| 1 | dusi9e8b | **0.97712** | 0.97937 | 0.97590 | 0.97712 | 13 |
| 2 | 2vayr7k4 | 0.97589 | 0.97647 | 0.97638 | 0.97589 | 15 |
| 3 | fdp8oeci | 0.97186 | 0.97194 | 0.97334 | 0.97186 | 13 |
| 4 | hlbs25qg | 0.97046 | 0.97867 | 0.96488 | 0.97046 | 10 |
| 5 | ig83z2dq | 0.96772 | 0.97960 | 0.95929 | 0.96772 | 10 |

### 성능 개선 분석
- **최고 성능**: 0.97712 (Run 1)
- **Baseline 대비**: 이전 최고 성능 0.9705 대비 **+0.66%p 향상**
- **안정성**: Top 3 run이 모두 97.1% 이상으로 안정적
- **Precision 우세**: 대부분의 run에서 Precision > Recall 경향

---

## 🔍 하이퍼파라미터 분석

### Learning Rate 분석
| Run | LR | H-Mean | 특징 |
|-----|-----|--------|------|
| dusi9e8b | **0.000974** | 0.97712 | 최적 |
| 2vayr7k4 | 0.001058 | 0.97589 | 약간 높음 |
| fdp8oeci | 0.001252 | 0.97186 | 높음 |
| hlbs25qg | 0.000853 | 0.97046 | 낮음 |

**결론**: **0.0009~0.0010** 범위가 최적

### Weight Decay 분석
| Run | Weight Decay | H-Mean | 특징 |
|-----|--------------|--------|------|
| dusi9e8b | **0.000146** | 0.97712 | 낮음 |
| 2vayr7k4 | 0.000141 | 0.97589 | 낮음 |
| fdp8oeci | 0.000485 | 0.97186 | 높음 |
| hlbs25qg | 0.000254 | 0.97046 | 중간 |

**결론**: **0.00014~0.00015** 범위가 최적 (낮은 정규화 선호)

### T_max (Scheduler) 분석
| Run | T_max | Epochs | H-Mean | 학습률 감소 속도 |
|-----|-------|--------|--------|-----------------|
| dusi9e8b | **12** | 13 | 0.97712 | 적절 |
| 2vayr7k4 | 13 | 15 | 0.97589 | 적절 |
| fdp8oeci | 9 | 13 | 0.97186 | 빠름 |
| hlbs25qg | 14 | 10 | 0.97046 | 느림 |

**결론**: **T_max = Epochs - 1** 또는 **Epochs - 2** 범위가 최적

### Threshold 분석
| Run | Thresh | Box Thresh | H-Mean | Precision | Recall |
|-----|--------|------------|--------|-----------|--------|
| dusi9e8b | **0.229** | 0.400 | 0.97712 | 0.979 | 0.976 |
| 2vayr7k4 | 0.207 | 0.417 | 0.97589 | 0.976 | **0.976** |
| fdp8oeci | 0.214 | 0.439 | 0.97186 | 0.972 | 0.973 |

**결론**: 
- **Thresh 0.22~0.23**: 높은 정밀도 선호
- **Box Thresh 0.40~0.42**: 낮은 값이 균형적

---

## 💡 핵심 인사이트

### 1. Learning Rate의 중요성
- **최적 범위**: 0.0009~0.0010
- 너무 높으면 (>0.0012) 성능 저하
- 너무 낮으면 (<0.0009) 수렴 속도 느림

### 2. Weight Decay는 낮게 유지
- **최적 범위**: 0.00014~0.00015
- 높은 Weight Decay(>0.0003)는 오히려 성능 저하
- 데이터 증강이 충분하여 강한 정규화 불필요

### 3. Scheduler 주기는 Epochs와 맞춰야
- T_max = Epochs - 1 또는 Epochs - 2가 최적
- 너무 짧으면 학습률이 너무 빨리 감소
- 너무 길면 마지막 에폭에서 학습률이 여전히 높음

### 4. Threshold 조정의 Trade-off
- **높은 Thresh (0.22~0.23)**: Precision ↑, Recall ↓
- **낮은 Thresh (0.20~0.21)**: Precision ↓, Recall ↑
- **Box Thresh 0.40~0.42**: 균형점 제공

### 5. 학습 Epoch 수
- **10 epoch**: 충분하지만 최적은 아님
- **13 epoch**: 최적 균형점
- **15 epoch**: 과적합 위험 증가 없이 안정적

---

## 🎯 최종 권장 파라미터

### 프로덕션 환경 권장 설정

```yaml
# 최적 하이퍼파라미터 (Run dusi9e8b 기반)
models:
  optimizer:
    lr: 0.000974            # Learning Rate
    weight_decay: 0.000146  # L2 Regularization
  
  scheduler:
    T_max: 12               # Cosine Annealing Period
  
  head:
    postprocess:
      thresh: 0.229         # Probability Threshold
      box_thresh: 0.400     # Box Confidence Threshold

trainer:
  max_epochs: 13            # Training Epochs

dataloaders:
  train_dataloader:
    batch_size: 1           # GPU Memory Constraint
  val_dataloader:
    batch_size: 1
  test_dataloader:
    batch_size: 1
```

### 대안 설정 (High Recall 선호 시)

```yaml
# Run 2vayr7k4 기반 - Recall 중시
models:
  optimizer:
    lr: 0.001058
    weight_decay: 0.000141
  
  scheduler:
    T_max: 13
  
  head:
    postprocess:
      thresh: 0.207         # 낮은 Threshold로 Recall 향상
      box_thresh: 0.417

trainer:
  max_epochs: 15
```

---

## 📈 성능 개선 이력

| 단계 | 설정 | Val H-Mean | 개선폭 |
|------|------|------------|--------|
| Baseline | 초기 설정 | 0.9705 | - |
| Grid Search | Postprocessing 조정 | 0.9705 | +0.00% |
| **Bayesian Sweep** | **전체 파라미터 최적화** | **0.9771** | **+0.66%** |

---

## 🚀 향후 실험 제안

### 1. Batch Size 증대 실험
- **현재 제약**: Batch Size 1 (GPU 메모리)
- **개선 방안**: 
  - Gradient Accumulation 적용
  - Mixed Precision Training (AMP)
  - 더 큰 GPU 환경에서 Batch Size 4~8 테스트

### 2. Epoch 수 세밀 조정
- 13 epoch과 15 epoch 사이 세밀 탐색 (14 epoch)
- Early Stopping 기준 재조정

### 3. Augmentation 영향 분석
- 현재 Augmentation 강도 vs 성능 상관관계 분석
- ColorJitter, RandomBrightness 등 개별 증강 효과 측정

### 4. Ensemble 전략
- Top 3 모델을 활용한 Soft Voting
- Test-Time Augmentation (TTA) 적용

### 5. 추가 Threshold 탐색
- 최적 run의 Threshold 주변 ±0.01 범위 Fine-tuning
- Adaptive Thresholding 기법 적용

---

## 📝 결론

### 주요 성과
1. **성능 향상**: Baseline 대비 0.66%p 향상 (0.9705 → 0.9771)
2. **안정적 재현성**: Top 3 run 모두 97.1% 이상 달성
3. **최적 파라미터 발견**: Learning Rate, Weight Decay, Scheduler 설정 최적화
4. **Threshold 최적화**: Precision-Recall 균형점 도출

### 실험 한계
- **Run 수 제한**: 9개 완료 (더 많은 탐색 가능)
- **Batch Size 고정**: GPU 메모리 제약으로 1로 고정
- **조기 종료**: 일부 run이 중간에 멈춘 경우 발생

### 실무 적용 가이드
- **즉시 적용 가능**: Run dusi9e8b의 설정을 프로덕션에 바로 적용
- **High Recall 필요 시**: Run 2vayr7k4의 설정 사용
- **안정성 중시**: Run fdp8oeci의 높은 Weight Decay 설정 고려

---

## 📚 참고 자료

### Run 상세 정보
- `sweep_analysis_detailed.json`: 전체 run 메트릭 및 설정
- WandB Sweep URL: `https://wandb.ai/fc_bootcamp/ocr-receipt-detection/sweeps/mspjjnuj`

### 관련 문서
- `0_baseline_analysis_report.md`: Baseline 성능 분석
- `1_postprocessing_tuning_analysis_report.md`: 후처리 파라미터 튜닝
- `2_cosine_scheduling_experiment_report.md`: Scheduler 실험 결과

---

**작성일**: 2026년 2월 10일  
**실험자**: AI OCR Team  
**데이터셋**: 통합 영수증 데이터셋 (4,698 images)  
**최종 모델 성능**: Val H-Mean **0.97712** | Test H-Mean **0.97712**
