#!/usr/bin/env python3
"""
Unclip Ratio Configuration Enhancement
Modify db_postprocess.py to make unclip_ratio configurable
"""

MODIFICATION_GUIDE = """
================================================================================
UNCLIP RATIO 설정 가능하도록 코드 수정 가이드
================================================================================

변경 대상 파일: ocr/models/head/db_postprocess.py

현재 문제:
- Line 140: polygon_unclip_ratio가 2.0으로 하드코딩
- Line 215: box_unclip_ratio 기본값 1.5로 하드코딩
- 설정 파일에서 조정 불가능

================================================================================
수정 방법
================================================================================

1. DBPostProcessor __init__ 메서드 수정
   (Line 22~28 수정)

변경 전:
```python
class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, use_polygon=False):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.use_polygon = use_polygon
```

변경 후:
```python
class DBPostProcessor:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, 
                 use_polygon=False,
                 box_unclip_ratio=1.5,        # 추가
                 polygon_unclip_ratio=2.0):    # 추가
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.use_polygon = use_polygon
        self.box_unclip_ratio = box_unclip_ratio      # 추가
        self.polygon_unclip_ratio = polygon_unclip_ratio  # 추가
```

================================================================================

2. polygons_from_bitmap 메서드 수정
   (Line 140 수정)

변경 전:
```python
# Unclip the box
if points.shape[0] > 2:
    box = self.unclip(points, unclip_ratio=2.0)  # ❌ 하드코딩
    if box is None:
        continue
```

변경 후:
```python
# Unclip the box
if points.shape[0] > 2:
    box = self.unclip(points, unclip_ratio=self.polygon_unclip_ratio)  # ✅ 설정값 사용
    if box is None:
        continue
```

================================================================================

3. boxes_from_bitmap 메서드 수정
   (Line 200 근처 수정)

변경 전:
```python
# Unclip the box
box = self.unclip(points).reshape(-1, 1, 2)  # ❌ 기본값만 사용
```

변경 후:
```python
# Unclip the box
box = self.unclip(points, unclip_ratio=self.box_unclip_ratio).reshape(-1, 1, 2)  # ✅ 설정값 사용
```

================================================================================

4. unclip 메서드 수정 (선택적)
   (Line 215~235)

현재 코드:
```python
def unclip(self, box, unclip_ratio=1.5):
    \"\"\"
    Expands the given box by a specified ratio.

    box: a list of points of shape (N, 2)
    unclip_ratio: the ratio of unclipping the box
    return: a list of points of shape (N, 2)
    \"\"\"
    # ... (기존 로직 유지)
```

→ 변경 불필요 (이미 파라미터로 받고 있음)

================================================================================
설정 파일 사용 예시
================================================================================

수정 후 configs/preset/models/head/db_head.yaml 에 추가 가능:

```yaml
models:
  head:
    _target_: ${head_path}.DBHead
    in_channels: 256
    upscale: 4
    k: 50
    bias: false
    smooth: false
    postprocess:
      thresh: 0.3
      box_thresh: 0.4
      max_candidates: 500
      use_polygon: true
      box_unclip_ratio: 1.5           # 추가 (box 모드용)
      polygon_unclip_ratio: 2.0       # 추가 (polygon 모드용)
```

================================================================================
테스트 명령어
================================================================================

# 1. Polygon unclip ratio 1.85 테스트 (High Precision)
python runners/predict.py \\
  checkpoint_path=outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt \\
  preset=hrnet_w44_1024 \\
  models.head.postprocess.polygon_unclip_ratio=1.85 \\
  exp_name=test_unclip_1.85

# 2. Polygon unclip ratio 2.0 테스트 (Current Baseline)
python runners/predict.py \\
  checkpoint_path=outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt \\
  preset=hrnet_w44_1024 \\
  models.head.postprocess.polygon_unclip_ratio=2.0 \\
  exp_name=test_unclip_2.0

# 3. Polygon unclip ratio 2.15 테스트 (High Recall)
python runners/predict.py \\
  checkpoint_path=outputs/hrnet_w44_1024_augmented_optimized/checkpoints/epoch=12-step=10634.ckpt \\
  preset=hrnet_w44_1024 \\
  models.head.postprocess.polygon_unclip_ratio=2.15 \\
  exp_name=test_unclip_2.15

================================================================================
예상 결과
================================================================================

| polygon_unclip_ratio | Recall | Precision | H-Mean | 전략 |
|---------------------|--------|-----------|---------|------|
| 1.85 | 0.973 ↓ | 0.987 ↑ | 0.9800 | High Precision |
| 2.0 (baseline) | 0.976 | 0.985 | 0.9806 | Balanced |
| 2.15 | 0.978 ↑ | 0.983 ↓ | 0.9805 | High Recall |

예상 개선: ±0.1~0.2%p (미미함)

================================================================================
"""

if __name__ == "__main__":
    print(MODIFICATION_GUIDE)
    
    print("\n" + "="*80)
    print("수정 적용 여부 확인")
    print("="*80)
    
    import os
    file_path = "ocr/models/head/db_postprocess.py"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            
        checks = {
            "box_unclip_ratio 파라미터": "box_unclip_ratio" in content.split('\n')[25],
            "polygon_unclip_ratio 파라미터": "polygon_unclip_ratio" in content.split('\n')[26],
            "self.box_unclip_ratio 속성": "self.box_unclip_ratio" in content,
            "self.polygon_unclip_ratio 속성": "self.polygon_unclip_ratio" in content,
        }
        
        print("\n현재 상태:")
        for check, result in checks.items():
            status = "✅ 적용됨" if result else "❌ 미적용"
            print(f"  {check}: {status}")
        
        if all(checks.values()):
            print("\n🎉 모든 수정이 적용되었습니다!")
        else:
            print("\n⚠️ 위 가이드대로 수정이 필요합니다.")
    else:
        print(f"\n❌ 파일을 찾을 수 없습니다: {file_path}")
        print("   baseline_code 디렉토리에서 실행하세요.")
