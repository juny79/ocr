# 실험 요약: 960px → 1024px 해상도 업그레이드

## 리더보드 최종 결과

| 메트릭 | 값 |
|--------|-----|
| **H-Mean** | **98.37%** ✅ |
| **Precision** | **98.18%** |
| **Recall** | **98.62%** |
| **제출 파일** | hrnet_w44_1024_submission_53.csv |

## 핵심 개선사항

### 1. 해상도 업그레이드
- 960×960 → **1024×1024** (+13.8% 픽셀)
- H-Mean: 97.80% → **98.37%** (+0.57%)
- GPU 메모리: 안정적 (Batch=4 유지)

### 2. 최적 하이퍼파라미터
```yaml
LR: 0.0001 → 0.001        (10배 증가)
T_max: 40 → 20             (코사인 사이클)
Max Epochs: 40 → 20        (50% 시간 단축)
Best Checkpoint: epoch=18  (3개 중 1위)
```

### 3. Config 구조 재설계
- 4가지 호환성 문제 해결
- DBTransforms/KeypointParams 정정
- DataLoader 구조 표준화

## 학습 결과

| 항목 | 값 |
|------|-----|
| 총 훈련 시간 | 3.5 시간 |
| 최고 에포크 | 18 (전체 20) |
| 내부 H-Mean | 98.59% |
| 리더보드 H-Mean | 98.37% |
| 격차 | -0.22% (분포 차이) |

## 아키텍처

```
(1024×1024)
  ↓
HRNet-W44 백본 (56.7M params)
  ↓
UNet 디코더
  ↓
DBHead
  ↓
: 텍스트 다각형
```

## 파일 구조

```
/data/ephemeral/home/
 3_hrnet_w44_1024_resolution_experiment_report.md  ← 상세 보고서
 baseline_code/outputs/hrnet_w44_1024/
   ├── checkpoints/epoch=18-step=15542.ckpt         ← 최적 모델
   └── submissions/hrnet_w44_1024_submission.csv    ← 제출 파일
 configs/preset/
    ├── db_augmented_1024.yaml
    ├── models/model_hrnet_w44_hybrid_1024.yaml
    └── hrnet_w44_1024.yaml
```

## 다음 단계 (향후 개선)

1. **외부 데이터 통합** (+1-2%)
   - SROIE, CORD-v2 병합
   
2. **K-fold 앙상블** (+0.5-1%)
   - 5-fold 교차검증
   
3. **고급 튜닝** (+0.3-0.7%)
   - WandB Sweep
   - 더 큰 해상도 실험

---

**최종 상태:** ✅ 98.37% H-Mean 달성 & 리더보드 검증 완료
