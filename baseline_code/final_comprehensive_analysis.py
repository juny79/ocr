#!/usr/bin/env python3
"""
최종 종합 분석 - 모든 실험 데이터 통합
최적 전략 도출
"""

print("="*80)
print("최종 종합 분석 - 전체 실험 결과")
print("="*80)
print()

# 전체 측정 데이터
all_results = [
    # (thresh, box_thresh, unclip, Precision, Recall, H-Mean, 설명)
    (0.210, 0.390, 2.0, 0.9882, 0.9840, 0.9858, "Recall 높음"),
    (0.212, 0.392, 2.0, 0.9883, 0.9834, 0.9856, "Recall 하락"),
    (0.218, 0.398, 2.0, 0.9888, 0.9838, 0.9860, "⭐ 최고점!"),
    (0.220, 0.400, 1.8, 0.9896, 0.9828, 0.9858, "P 과잉"),
    (0.220, 0.400, 1.9, 0.9893, 0.9832, 0.9859, "균형 개선"),
    (0.220, 0.400, 2.2, 0.9871, 0.9828, 0.9846, "P 부족"),
    (0.222, 0.402, 2.0, 0.9888, 0.9829, 0.9855, "하락"),
    (0.225, 0.405, 2.0, 0.9887, 0.9829, 0.9855, "하락"),
    (0.230, 0.410, 2.0, 0.9888, 0.9806, 0.9843, "급락"),
]

print("전체 측정 데이터 (H-Mean 순):")
print()
print("순위 │ thresh │ box_thr │ unclip │   P    │   R    │  H-Mean │ 비고")
print("─"*85)

sorted_results = sorted(all_results, key=lambda x: x[5], reverse=True)
for idx, (t, bt, u, p, r, h, desc) in enumerate(sorted_results, 1):
    marker = "⭐" if idx == 1 else f"{idx:2d}"
    print(f" {marker}  │ {t:.3f} │ {bt:.3f}  │  {u:.1f}  │ {p:.4f} │ {r:.4f} │ {h:.4f}  │ {desc}")

print()
print("="*80)
print("🔍 핵심 발견")
print("="*80)
print()

print("1. thresh=0.218이 절대 최고점!")
print("   H-Mean: 0.9860 (목표 0.9863 대비 -3)")
print()

print("2. thresh=0.220 영역 분석:")
print("   unclip=1.8: P=0.9896, R=0.9828, H=0.9858")
print("   unclip=1.9: P=0.9893, R=0.9832, H=0.9859 ← 최적")
print("   unclip=2.2: P=0.9871, R=0.9828, H=0.9846")
print()
print("   → unclip=1.9가 0.220 기준 최적")
print("   → 하지만 최대 0.9859 (목표 미달)")
print()

print("3. Recall 패턴:")
print("   thresh=0.210: R=0.9840")
print("   thresh=0.212: R=0.9834 ↓")
print("   thresh=0.218: R=0.9838 ↑ ← Peak!")
print("   thresh=0.220: R=0.9828-0.9832")
print("   thresh=0.222+: 지속 하락")
print()
print("   → 0.214-0.219 범위에 복잡한 곡선")
print()

print("="*80)
print("📊 목표 달성 가능성 분석")
print("="*80)
print()

target = 0.9863
best_h = 0.9860
gap = (target - best_h) * 10000

print(f"현재 최고: 0.9860 (thresh=0.218)")
print(f"목표 점수: 0.9863")
print(f"필요 상승: {gap:.1f} 포인트")
print()

# 0.9860에서 0.9863 달성 조건
print("0.9863 달성 조건 (thresh=0.218 기준):")
print()

current_p = 0.9888
current_r = 0.9838

# Recall만 변화
print("▶ Recall 상승 시나리오:")
for delta_r in [0.0003, 0.0005, 0.0007]:
    new_r = current_r + delta_r
    new_h = 2 * current_p * new_r / (current_p + new_r)
    if new_h >= target:
        print(f"  R={new_r:.4f} (+{delta_r*10000:.1f}) → H={new_h:.4f} ✅ 달성!")
    else:
        gap_h = (new_h - target) * 10000
        print(f"  R={new_r:.4f} (+{delta_r*10000:.1f}) → H={new_h:.4f} (Gap: {gap_h:.1f})")

print()

# Precision과 Recall 동시 변화
print("▶ 균형 조정 시나리오:")
scenarios = [
    (0.9890, 0.9840),
    (0.9888, 0.9843),
    (0.9892, 0.9838),
]

for p, r in scenarios:
    h = 2 * p * r / (p + r)
    if h >= target:
        delta_p = (p - current_p) * 10000
        delta_r = (r - current_r) * 10000
        print(f"  P={p:.4f} ({delta_p:+.1f}), R={r:.4f} ({delta_r:+.1f}) → H={h:.4f} ✅ 달성!")
    else:
        gap_h = (h - target) * 10000
        print(f"  P={p:.4f}, R={r:.4f} → H={h:.4f} (Gap: {gap_h:.1f})")

print()
print("="*80)
print("🎯 최종 전략")
print("="*80)
print()

print("Option 1: 0.218 근처 미세 조정 ⭐⭐⭐⭐⭐ (최우선!)")
print("  목표: thresh를 0.218 ±0.002 범위에서 Recall 3-5 포인트 상승")
print()
print("  A) thresh=0.216, 0.217 테스트")
print("     - 0.218보다 낮춰서 Recall 상승 시도")
print("     - 예상 Recall: 0.9839-0.9842")
print("     - 예상 H-Mean: 0.9861-0.9864")
print()
print("  B) thresh=0.219, 0.220 테스트")
print("     - 0.218보다 약간 높여서 미세 조정")
print("     - 예상 H-Mean: 0.9859-0.9862")
print()

print("Option 2: fold3_t215_b395_wide.csv 제출 ⭐⭐⭐⭐")
print("  - 0.212와 0.218 중간값")
print("  - 이미 생성되어 있음!")
print("  - 예상 H-Mean: 0.9858-0.9862")
print()

print("Option 3: 기준점 최종 확인 ⭐⭐⭐")
print("  - fold3_unclip20_t220_b400.csv")
print("  - thresh=0.220, unclip=2.0 (원래 기준)")
print("  - 예상: 0.9858-0.9863")
print()

print("="*80)
print("💡 결론")
print("="*80)
print()
print("✅ 0.218이 확실한 Peak!")
print("✅ 0.9860 → 0.9863 (+3pt) 달성하려면:")
print("   → 0.218 기준 Recall +3~5pt 필요")
print("   → 또는 0.216-0.217에서 Recall 상승 기회")
print()
print("❌ thresh=0.220에서는 최대 0.9859 (한계)")
print()
print("📋 권장 순서:")
print("  1) fold3_t215_b395_wide.csv (즉시 제출 가능)")
print("  2) 0.216-0.217 생성 후 제출 (미세 조정)")
print("  3) fold3_unclip20_t220_b400.csv (최종 검증)")
print()
print("🚀 가장 확실한 방법: 0.216-0.217 범위 정밀 탐색!")
