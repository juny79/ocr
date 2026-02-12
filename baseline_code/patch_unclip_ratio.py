#!/usr/bin/env python3
"""
DBPostProcessor에 unclip_ratio config 지원 추가
코드 패치 스크립트
"""
import sys

POSTPROCESS_FILE = '/data/ephemeral/home/baseline_code/ocr/models/head/db_postprocess.py'

def patch_code():
    """db_postprocess.py 코드 패치"""
    
    with open(POSTPROCESS_FILE, 'r') as f:
        content = f.read()
    
    # 백업
    with open(POSTPROCESS_FILE + '.backup', 'w') as f:
        f.write(content)
    
    print("✓ 백업 생성: db_postprocess.py.backup")
    
    # 1. __init__ 파라미터 추가
    old_init = 'def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, use_polygon=False):'
    new_init = 'def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, use_polygon=False, unclip_ratio=2.0):'
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✓ __init__ 파라미터 수정")
    else:
        print("❌ __init__ 시그니처를 찾을 수 없습니다")
        return False
    
    # 2. self.unclip_ratio 저장 추가
    old_assignments = '''        self.min_size = 3                       # minimum size of text region
        self.thresh = thresh                    # threshold for binarization
        self.box_thresh = box_thresh            # threshold for text region proposals
        self.max_candidates = max_candidates    # max number of text region proposals
        self.use_polygon = use_polygon          # use polygon or box'''
    
    new_assignments = '''        self.min_size = 3                       # minimum size of text region
        self.thresh = thresh                    # threshold for binarization
        self.box_thresh = box_thresh            # threshold for text region proposals
        self.max_candidates = max_candidates    # max number of text region proposals
        self.use_polygon = use_polygon          # use polygon or box
        self.unclip_ratio = unclip_ratio        # unclip ratio for polygon expansion'''
    
    if old_assignments in content:
        content = content.replace(old_assignments, new_assignments)
        print("✓ self.unclip_ratio 변수 추가")
    else:
        print("❌ 변수 할당 코드를 찾을 수 없습니다")
        return False
    
    # 3. polygons_from_bitmap의 하드코딩 수정
    old_unclip_poly = 'box = self.unclip(points, unclip_ratio=2.0)'
    new_unclip_poly = 'box = self.unclip(points, unclip_ratio=self.unclip_ratio)'
    
    if old_unclip_poly in content:
        content = content.replace(old_unclip_poly, new_unclip_poly)
        print("✓ polygons_from_bitmap 수정 (하드코딩 제거)")
    else:
        print("⚠️  polygons_from_bitmap 하드코딩을 찾을 수 없습니다 (이미 수정됨?)")
    
    # 4. boxes_from_bitmap의 unclip도 수정
    old_unclip_box = 'box = self.unclip(points).reshape(-1, 1, 2)'
    new_unclip_box = 'box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)'
    
    if old_unclip_box in content:
        content = content.replace(old_unclip_box, new_unclip_box)
        print("✓ boxes_from_bitmap 수정")
    else:
        print("⚠️  boxes_from_bitmap 수정 스킵 (이미 수정됨?)")
    
    # 파일 쓰기
    with open(POSTPROCESS_FILE, 'w') as f:
        f.write(content)
    
    print("\n✅ 코드 패치 완료!")
    print(f"   파일: {POSTPROCESS_FILE}")
    print(f"   백업: {POSTPROCESS_FILE}.backup")
    print()
    print("이제 config에서 다음과 같이 설정 가능합니다:")
    print("  models.head.postprocess.unclip_ratio: 1.8  # 또는 2.0, 2.2 등")
    
    return True

def restore_backup():
    """백업 복원"""
    import shutil
    backup = POSTPROCESS_FILE + '.backup'
    try:
        shutil.copy(backup, POSTPROCESS_FILE)
        print(f"✓ 백업 복원 완료: {backup} → {POSTPROCESS_FILE}")
    except FileNotFoundError:
        print(f"❌ 백업 파일 없음: {backup}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'restore':
        restore_backup()
    else:
        print("="*80)
        print("DBPostProcessor Unclip Ratio Config 지원 패치")
        print("="*80)
        print()
        success = patch_code()
        if success:
            print("\n다음 명령으로 복원 가능:")
            print("  python patch_unclip_ratio.py restore")
