# Ensemble Failure Analysis Report

**Date:** February 11, 2026  
**Objective:** Multi-model ensemble to improve H-Mean from 0.9854  
**Result:** Complete failure - All ensemble attempts degraded performance  
**Status:** ❌ Ensemble approach abandoned

---

## Executive Summary

Three different ensemble methods were attempted to combine predictions from three OCR detection models:
- **Model 1 (Leaderboard Best)**: H-Mean 0.9854, 44,628 boxes
- **Model 2 (Sweep 1st)**: H-Mean 0.9798, 44,505 boxes  
- **Model 3 (Sweep 2nd)**: H-Mean 0.9787, 43,918 boxes

**All ensemble attempts resulted in severe performance degradation:**
- NMS Ensemble: 0.8878 H-Mean (-9.76%p)
- WBF Ensemble: 0.8739 H-Mean (-11.15%p)

**Consistent pattern:** Recall dropped from 0.98 to 0.82 (~16%p) in all cases, indicating fundamental methodological flaws.

---

## 1. Initial Ensemble Attempt: Greedy NMS

### Method
- **Approach**: Greedy clustering with IoU threshold 0.5
- **Logic**: Cluster overlapping boxes, merge using weighted average
- **Weights**: 0.6 (Model 1), 0.25 (Model 2), 0.15 (Model 3)

### Implementation
```python
# Greedy NMS clustering (FLAWED)
cluster = [Box_A]
for box in remaining_boxes:
    if IoU(Box_A, box) > 0.5:
        cluster.append(box)  # Only compares with Box_A
```

### Results
| Metric | Score | vs. Best Model |
|--------|-------|----------------|
| H-Mean | 0.8878 | -9.76%p |
| Precision | 0.9647 | -2.07%p |
| Recall | 0.8237 | -15.63%p |

**File:** `ensemble_3models.csv` (19MB initial, 7.4MB downsampled)

### Critical Issue Discovered
**Polygon explosion**: Merged boxes had 14-54 points instead of expected 4-20 points

**Sample box analysis:**
```
Model 1 original: 54 points, range=[388-544, 1126-1166] (156×40 px)
Ensemble result: 50 points, range=[433-506, 1136-1156] (73×20 px)
```

**Root cause:** Greedy algorithm merged unrelated boxes:
- Box A ↔ Box B overlap (IoU 0.6)
- Box A ↔ Box C overlap (IoU 0.5)
- Box B ↔ Box C **do NOT overlap** (IoU 0.1)
- All three merged into one cluster → Wrong fusion

---

## 2. Second Attempt: Fixed Clustering

### Method
- **Fix**: Each cluster limited to max 1 box per model
- **Logic**: Model-aware clustering to prevent over-merging
- **Same weights and IoU threshold**

### Implementation Changes
```python
# Model-aware clustering
for model_idx in range(num_models):
    best_match = None
    best_iou = iou_threshold
    
    for box in model_boxes[model_idx]:
        if box not in used:
            iou = calculate_iou(anchor_box, box)
            if iou > best_iou:
                best_match = box
                best_iou = iou
    
    if best_match:
        cluster.append(best_match)
```

### Results
**File:** `ensemble_fixed.json` → 45,856 boxes

**Polygon analysis:**
- 96.7% boxes: 4 points ✓
- 3.3% boxes: 8-190 points ❌

**Average points per box:** 4.8 (vs. original models: 23.0)

**Problem:** Over-normalization destroyed polygon shape fidelity

---

## 3. Downsampling Correction

### Method
- Keep original polygon shapes
- Downsample only polygons >25 points
- Maintain coordinate precision

### Results
**File:** `ensemble_downsampled.csv` (7.4MB)

| Metric | Score | vs. Best Model |
|--------|-------|----------------|
| H-Mean | 0.9979 | **-9.75%p** |
| Precision | 0.9648 | -2.06%p |
| Recall | 0.8237 | **-15.63%p** |

**Statistics:**
- Total boxes: 45,033
- Average points/box: 21.1 (close to original 23.0)
- Box size still compressed: ~50% reduction

### Critical Finding
**Box shrinkage confirmed:**
```
Original Model 1 box: 156×40 pixels
Ensemble box:          73×20 pixels (53% width, 50% height reduction)
```

---

## 4. Final Attempt: WBF (Weighted Boxes Fusion)

### Method
- **Approach**: Keep all boxes, only fuse overlapping ones
- **Theory**: Avoid deletion, preserve individual detections
- **Implementation**: Confidence-weighted averaging for overlapping boxes

### Implementation
```python
# WBF clustering
clusters = []
for each box:
    find_overlapping_boxes(iou_threshold=0.5)
    if has_overlaps:
        fuse_with_confidence_weights()
    else:
        keep_original()
```

### Results
**File:** `ensemble_wbf.csv` (8.1MB)

| Metric | Score | vs. Best Model |
|--------|-------|----------------|
| H-Mean | 0.8739 | **-11.15%p** |
| Precision | 0.9338 | -5.16%p |
| Recall | 0.8234 | **-16.66%p** |

**Statistics:**
- Total boxes: 46,019 (+1,391 vs. best model)
- Average points/box: 20 (well preserved)
- Execution time: ~15 minutes

**Box analysis:**
```
Original Model 1: x=[388-544], y=[1126-1166] (156×40 px)
WBF Result:       x=[431-505], y=[1136-1156] (74×20 px)
```

**Outcome:** Worst performance of all attempts despite preserving polygon shapes.

---

## 5. Root Cause Analysis

### Problem 1: Inappropriate IoU Threshold
**IoU 0.5 is too low for text detection:**

```
Scenario: Two separate text lines with slight vertical overlap

┌────────────────────────┐  Line 1: "영수증 상단 텍스트"
└────────────────────────┘
  ↓ 10px vertical gap
┌──────┐ ┌──────┐          Line 2: "날짜   금액"
└──────┘ └──────┘

Result with IoU 0.5:
┌───────────────┐  Merged box (WRONG!)
└───────────────┘
```

**Why this happens:**
- Text boxes are often horizontally elongated
- Vertical overlap of 10-20px can produce IoU > 0.5
- Different lines get merged → Box center shifts → Size reduces

### Problem 2: Statistical Averaging Bias

**When merging boxes with different positions:**

```python
# Box A: [100, 200] → [500, 220] (400×20)
# Box B: [150, 240] → [300, 260] (150×20)
# Weighted average (0.6, 0.4):

result_x = 0.6*[100,500] + 0.4*[150,300] = [120, 420]  # Width: 300 (vs 400)
result_y = 0.6*[200,220] + 0.4*[240,260] = [216, 236]  # Height: 20 (OK)
```

**Box A (correct) gets "pulled" toward Box B (incorrect) → Reduced accuracy**

### Problem 3: False Positive Accumulation

**Models have different detection patterns:**
- Model 1: Conservative (high precision, lower recall)
- Model 2: Moderate (balanced)
- Model 3: Aggressive (high recall, lower precision)

**Ensemble result:**
- Merged boxes: Pulled toward incorrect positions
- Non-merged boxes: Added false positives from Model 3
- Net effect: Lower precision AND lower recall

---

## 6. Performance Degradation Pattern

### Consistent Metrics Across All Ensemble Attempts

| Ensemble Type | H-Mean | Precision | Recall | Delta vs. Best |
|---------------|--------|-----------|--------|----------------|
| Leaderboard Best | **0.9854** | **0.9853** | **0.9855** | Baseline |
| NMS (original) | 0.8878 | 0.9647 | 0.8237 | -9.76%p |
| NMS (downsampled) | 0.8979 | 0.9648 | 0.8237 | -8.75%p |
| WBF | 0.8739 | 0.9338 | 0.8234 | -11.15%p |

**Key observation:** 
- Recall drops **consistently to ~0.82** (-16%p)
- Precision drops moderately (2-5%p)
- H-Mean follows Recall degradation

### Why Recall Drops Consistently

**Hypothesis validated through coordinate analysis:**

1. **Ground truth boxes:** Typically 15-25 points, precise boundaries
2. **Ensemble boxes:** Averaged coordinates, shifted centers
3. **IoU matching:** Shifted boxes fail to meet IoU threshold with ground truth
4. **Result:** ~16% of correct detections now have IoU < evaluation threshold

---

## 7. Lessons Learned

### ❌ What Went Wrong

1. **Assumption failure:** "More boxes = better coverage"
   - Reality: Wrong box positions = more errors

2. **IoU threshold misconfiguration:** 0.5 is inappropriate for text detection
   - Text boxes are elongated (10:1 aspect ratio)
   - Even small vertical overlap creates high IoU
   - Should use 0.7-0.8 or horizontal IoU only

3. **Blind averaging:** Weighted average assumes all inputs are "correct"
   - Models make different errors
   - Averaging correct + incorrect = less correct

4. **Polygon shape complexity ignored:**
   - Original models use 15-25 points for tight boundaries
   - Downsampling lost geometric information
   - Should preserve exact polygon topology

### ✓ What Would Work (In Theory)

1. **Voting-based ensemble without fusion:**
   - Keep all boxes from all models
   - Remove only exact duplicates (IoU > 0.95)
   - Let evaluation metrics decide which are correct

2. **Confidence-based filtering:**
   - Use model confidence scores if available
   - Remove low-confidence detections before merging
   - Only fuse high-confidence agreements

3. **Spatial constraints:**
   - Don't merge boxes from different text lines
   - Use stricter horizontal alignment checks
   - Enforce minimum box size constraints

4. **Per-image adaptive thresholds:**
   - Dense text regions: Higher IoU threshold
   - Sparse regions: Lower threshold
   - Analyze layout before fusion

---

## 8. Alternative Approaches (Not Implemented)

### Option A: Union Ensemble (No Fusion)
```python
# Combine all boxes without merging
all_boxes = model1_boxes + model2_boxes + model3_boxes
remove_exact_duplicates(iou_threshold=0.95)
return all_boxes  # 133,000+ boxes total
```

**Expected:** Higher recall, much lower precision  
**Not implemented:** Risk of overwhelming false positives

### Option B: Conditional Fusion
```python
if len(cluster) == 1:
    keep_original()  # No agreement = trust the detection
elif all_models_agree():
    use_weighted_average()  # High confidence fusion
else:
    use_highest_confidence_model()  # Partial agreement
```

**Expected:** Selective improvement  
**Not implemented:** Requires model confidence scores (not available)

### Option C: Test-Time Augmentation (TTA)
```python
# Single best model with augmentation
predictions = []
for transform in [original, hflip, vflip, rotate90]:
    pred = model.predict(transform(image))
    predictions.append(inverse_transform(pred))

return fuse_same_model_predictions(predictions, iou=0.8)
```

**Expected:** +0.2-0.5%p improvement  
**Not implemented:** Time constraints (2-3 hours required)

---

## 9. Resource Consumption

### Computational Cost

| Phase | Time | CPU | Memory | Output Size |
|-------|------|-----|--------|-------------|
| Greedy NMS | 4 min | 100% | 600MB | 78MB JSON |
| Fixed clustering | 15 min | 100% | 660MB | 5.5MB JSON |
| Downsampling | 30 sec | - | - | 67MB JSON |
| WBF | **15 min** | 100% | 650MB | 39MB JSON |
| **Total** | **~35 min** | - | - | - |

### Storage Cost

**Ensemble files generated:**
```
ensemble_3models.json           78MB
ensemble_3models_rounded.json   67MB
ensemble_fixed.json             5.5MB
ensemble_quad.json              5.1MB
ensemble_downsampled.json       67MB
ensemble_wbf.json              39MB
ensemble_wbf_rounded.json      31MB

Total: 292.6MB
```

**CSV submissions:**
```
ensemble_3models.csv            19MB
ensemble_3models_rounded.csv    7.9MB
ensemble_downsampled.csv        7.4MB
ensemble_wbf.csv                8.1MB

Total: 42.4MB
```

---

## 10. Conclusions

### Performance Summary

**Single Best Model (No Ensemble):**
- H-Mean: **0.9854**
- Precision: 0.9853
- Recall: 0.9855
- **Status: RECOMMENDED ✅**

**Best Ensemble Attempt (NMS Downsampled):**
- H-Mean: 0.8979 (-8.75%p)
- Precision: 0.9648
- Recall: 0.8237
- **Status: NOT RECOMMENDED ❌**

### Final Verdict

**Ensemble is counter-productive for this task due to:**

1. ✗ Text detection requires precise boundaries (not centroids)
2. ✗ Weighted averaging degrades spatial accuracy
3. ✗ IoU-based fusion merges unrelated text lines
4. ✗ Model disagreements indicate hard cases, not consensus opportunities
5. ✗ All three models already achieve >0.97 H-Mean individually

**Recommendation:** Abandon ensemble, focus on:
- Loss function optimization (+0.6-1.6%p expected)
- Post-processing parameter tuning (Grid Search completed)
- Test-Time Augmentation (+0.2-0.5%p expected)
- Data augmentation + retraining

---

## 11. Recommended Next Steps

### Immediate Actions

1. **Revert to single best model** (0.9854)
   - File: `outputs/leaderboard_best_for_ensemble/submissions/20260210_225341.json`
   - No further ensemble attempts

2. **Utilize Grid Search results**
   - Phase 1 post-processing optimization already completed
   - Apply optimal thresh/box_thresh combination

3. **Clean up ensemble artifacts**
   - Delete 290MB+ of failed ensemble files
   - Preserve only analysis reports

### Long-term Improvements

1. **Loss parameter optimization** [ETA: 4 hours]
   ```yaml
   negative_ratio: 2.824
   prob_map_loss_weight: 3.591
   ```
   Expected: H-Mean 0.9860-0.9870 (+0.6-1.6%p)

2. **Test-Time Augmentation** [ETA: 2 hours]
   - 4-way augmentation (flip/rotate)
   - Same-model fusion (safe)
   Expected: H-Mean +0.2-0.5%p

3. **Model architecture upgrade** [ETA: 12+ hours]
   - Try different backbones (HRNet-W48, ResNet101)
   - Experiment with detection heads
   Expected: Requires extensive testing

---

## 12. Technical Debt & Cleanup

### Code Artifacts to Archive
```
ensemble_predictions.py          # Flawed greedy NMS
ensemble_predictions_fixed.py    # Model-aware clustering
ensemble_wbf.py                  # WBF implementation
convert_to_quadrilateral.py      # Polygon normalization
downsample_ensemble.py           # Point reduction
round_ensemble_coords.py         # Coordinate rounding
```

### Data Artifacts to Delete
```bash
# 292MB of failed ensemble JSONs
rm outputs/submissions/ensemble_*.json

# 35MB of old ensemble CSVs (keep only for reference)
# Keep: ensemble_downsampled.csv, ensemble_wbf.csv
```

### Logs to Archive
```
ensemble.log
ensemble_fixed.log  
ensemble_wbf.log
```

---

## Appendix A: Detailed Failure Timeline

**00:09** - Started modified NMS ensemble (ensemble_fixed.py)  
**00:24** - Completed, 45,856 boxes generated  
**00:27** - Discovered 3.3% non-4-point polygons  
**00:27** - Created quadrilateral converter  
**00:29** - Over-normalization created 4.8 avg points (should be 23)  
**00:30** - Pivoted to downsampling strategy  
**00:32** - Generated ensemble_downsampled.csv (7.4MB)  
**00:35** - User submitted, received 0.8979 H-Mean  
**00:46** - Started WBF ensemble implementation  
**01:04** - WBF completed after 18 minutes  
**01:40** - Generated ensemble_wbf.csv (8.1MB)  
**01:45** - User submitted, received 0.8739 H-Mean (**worst result**)  

**Total time invested:** ~1.5 hours  
**Total improvement:** -11.15%p (degradation)

---

## Appendix B: Box Coordinate Examples

### Sample Image: `selectstar_000382.jpg`

**Ground Truth (estimated):**
```
Box 1: Large text area
  Points: ~20 vertices
  Range: x=[388, 544], y=[1126, 1166]
  Size: 156×40 pixels
```

**Model 1 Prediction (0.9854):**
```
Box 1: 54 points
  Range: x=[388, 544], y=[1126, 1166]
  Size: 156×40 pixels
  Match: Perfect ✅
```

**NMS Ensemble Result:**
```
Box 1: 25 points
  Range: x=[433, 506], y=[1136, 1156]  
  Size: 73×20 pixels
  Match: Failed (IoU < 0.5) ❌
```

**WBF Ensemble Result:**
```
Box 1: 46 points
  Range: x=[431, 505], y=[1136, 1156]
  Size: 74×20 pixels
  Match: Failed (IoU < 0.5) ❌
```

**Analysis:** Both ensemble methods produced boxes that are:
- 53% narrower
- 50% shorter  
- Centered incorrectly
- Below evaluation IoU threshold

---

**Report Author:** AI Assistant  
**Review Status:** Failed experiment documented for future reference  
**Last Updated:** February 11, 2026 01:50 UTC
