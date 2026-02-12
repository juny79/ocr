"""
Validation Set Analysis & Challenging Cases
최고 성능 모델 분석용 - Validation Set 통계 및 어려운 케이스 식별
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


def load_val_data(val_json_path):
    """Load validation set"""
    with open(val_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['images']
    return data


def analyze_box_characteristics(gt_data):
    """Analyze text box characteristics"""
    results = []
    
    for img_name, img_info in gt_data.items():
        boxes_data = []
        for word_id, word_info in img_info['words'].items():
            points = np.array(word_info['points'])
            
            # Calculate box properties
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            width = x_coords.max() - x_coords.min()
            height = y_coords.max() - y_coords.min()
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            boxes_data.append({
                'width': width,
                'height': height,
                'area': area,
                'aspect_ratio': aspect_ratio
            })
        
        # Image-level statistics
        if boxes_data:
            df_boxes = pd.DataFrame(boxes_data)
            results.append({
                'image_name': img_name,
                'num_boxes': len(boxes_data),
                'img_width': img_info['img_w'],
                'img_height': img_info['img_h'],
                'img_area': img_info['img_w'] * img_info['img_h'],
                'avg_box_width': df_boxes['width'].mean(),
                'avg_box_height': df_boxes['height'].mean(),
                'avg_box_area': df_boxes['area'].mean(),
                'min_box_area': df_boxes['area'].min(),
                'max_box_area': df_boxes['area'].max(),
                'avg_aspect_ratio': df_boxes['aspect_ratio'].mean(),
                'density': len(boxes_data) / (img_info['img_w'] * img_info['img_h']) * 1e6  # boxes per million pixels
            })
    
    return pd.DataFrame(results)


def identify_challenging_cases(df):
    """Identify potentially challenging images"""
    challenging = []
    
    # High box count
    high_count = df.nlargest(20, 'num_boxes')
    for _, row in high_count.iterrows():
        challenging.append({
            'image_name': row['image_name'],
            'challenge_type': 'High Box Count',
            'value': row['num_boxes'],
            'description': f'{int(row["num_boxes"])} boxes'
        })
    
    # High density
    high_density = df.nlargest(20, 'density')
    for _, row in high_density.iterrows():
        challenging.append({
            'image_name': row['image_name'],
            'challenge_type': 'High Density',
            'value': row['density'],
            'description': f'{row["density"]:.1f} boxes/Mpx'
        })
    
    # Very small boxes
    small_boxes = df.nsmallest(20, 'min_box_area')
    for _, row in small_boxes.iterrows():
        challenging.append({
            'image_name': row['image_name'],
            'challenge_type': 'Small Boxes',
            'value': row['min_box_area'],
            'description': f'Min area: {row["min_box_area"]:.0f} px²'
        })
    
    # Extreme aspect ratios
    extreme_ar = df.nlargest(20, 'avg_aspect_ratio')
    for _, row in extreme_ar.iterrows():
        challenging.append({
            'image_name': row['image_name'],
            'challenge_type': 'Extreme Aspect Ratio',
            'value': row['avg_aspect_ratio'],
            'description': f'Avg AR: {row["avg_aspect_ratio"]:.2f}'
        })
    
    return pd.DataFrame(challenging)


def create_visualizations(df, output_dir):
    """Create analysis visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Box count distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(df['num_boxes'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Boxes', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Box Counts', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(df['num_boxes'].median(), color='red', linestyle='--', label=f'Median: {df["num_boxes"].median():.0f}')
    axes[0, 0].legend()
    
    # 2. Image size distribution
    axes[0, 1].scatter(df['img_width'], df['img_height'], alpha=0.5)
    axes[0, 1].set_xlabel('Image Width', fontsize=12)
    axes[0, 1].set_ylabel('Image Height', fontsize=12)
    axes[0, 1].set_title('Image Size Distribution', fontsize=14, fontweight='bold')
    
    # 3. Box density
    axes[1, 0].hist(df['density'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Density (boxes/Mpx)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Text Box Density Distribution', fontsize=14, fontweight='bold')
    
    # 4. Average box area
    axes[1, 1].hist(np.log10(df['avg_box_area']), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Log10(Average Box Area)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Box Area Distribution (log scale)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'validation_statistics.png'}")
    
    # 2. Box count vs image size
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['img_area']/1e6, df['num_boxes'], 
                         c=df['density'], cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('Image Area (Megapixels)', fontsize=12)
    ax.set_ylabel('Number of Boxes', fontsize=12)
    ax.set_title('Box Count vs Image Size (colored by density)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density (boxes/Mpx)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'box_count_vs_image_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'box_count_vs_image_size.png'}")


def main():
    VAL_JSON = '/data/ephemeral/home/data/datasets/jsons/val.json'
    OUTPUT_DIR = Path('/data/ephemeral/home/error_analysis')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("VALIDATION SET COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\nValidation JSON: {VAL_JSON}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Load data
    print("\nLoading validation set...")
    gt_data = load_val_data(VAL_JSON)
    print(f"  - Loaded {len(gt_data)} images")
    
    # Analyze characteristics
    print("\nAnalyzing box characteristics...")
    df = analyze_box_characteristics(gt_data)
    
    # Save detailed stats
    df.to_csv(OUTPUT_DIR / 'validation_detailed_stats.csv', index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'validation_detailed_stats.csv'}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    print(f"\n{'Dataset Overview':^80}")
    print("-"*80)
    print(f"  Total Images:              {len(df):,}")
    print(f"  Total Boxes:               {df['num_boxes'].sum():,}")
    print(f"  Average Boxes per Image:   {df['num_boxes'].mean():.2f}")
    print(f"  Median Boxes per Image:    {df['num_boxes'].median():.0f}")
    print(f"  Min Boxes:                 {df['num_boxes'].min()}")
    print(f"  Max Boxes:                 {df['num_boxes'].max()}")
    
    print(f"\n{'Image Sizes':^80}")
    print("-"*80)
    print(f"  Avg Width:                 {df['img_width'].mean():.0f} px")
    print(f"  Avg Height:                {df['img_height'].mean():.0f} px")
    print(f"  Avg Area:                  {df['img_area'].mean()/1e6:.2f} Mpx")
    
    print(f"\n{'Box Characteristics':^80}")
    print("-"*80)
    print(f"  Avg Box Width:             {df['avg_box_width'].mean():.1f} px")
    print(f"  Avg Box Height:            {df['avg_box_height'].mean():.1f} px")
    print(f"  Avg Box Area:              {df['avg_box_area'].mean():.0f} px²")
    print(f"  Avg Density:               {df['density'].mean():.2f} boxes/Mpx")
    print(f"  Avg Aspect Ratio:          {df['avg_aspect_ratio'].mean():.2f}")
    
    # Identify challenging cases
    print("\n" + "="*80)
    print("POTENTIALLY CHALLENGING CASES")
    print("="*80)
    
    challenging_df = identify_challenging_cases(df)
    
    by_type = challenging_df.groupby('challenge_type')
    for challenge_type, group in by_type:
        print(f"\n{challenge_type}:")
        print("-"*80)
        top_cases = group.nlargest(10, 'value')
        for idx, row in top_cases.iterrows():
            print(f"  {row['image_name']:60s} | {row['description']}")
    
    challenging_df.to_csv(OUTPUT_DIR / 'challenging_cases.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'challenging_cases.csv'}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    create_visualizations(df, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  - validation_detailed_stats.csv     (모든 이미지 통계)")
    print(f"  - challenging_cases.csv             (어려운 케이스 목록)")
    print(f"  - validation_statistics.png         (통계 분포 시각화)")
    print(f"  - box_count_vs_image_size.png       (박스 수와 이미지 크기 관계)")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR ERROR ANALYSIS")
    print("="*80)
    print("\n모델 에러 분석을 위한 권장 사항:")
    print(f"\n1. 고밀도 이미지 ({df['density'].quantile(0.9):.1f}+ boxes/Mpx)")
    print(f"   - 예측 누락 (False Negative) 위험이 높음")
    print(f"   - 상위 20개 케이스 확인 필요")
    
    print(f"\n2. 다수 박스 이미지 ({df['num_boxes'].quantile(0.9):.0f}+ boxes)")
    print(f"   - 복잡한 레이아웃으로 오탐 (False Positive) 가능성")
    print(f"   - 박스 매칭 어려움")
    
    print(f"\n3. 소형 박스 이미지 (min area < {df['min_box_area'].quantile(0.1):.0f} px²)")
    print(f"   - 작은 텍스트 영역 검출 실패 가능")
    print(f"   - Threshold 조정 필요 가능성")
    
    print(f"\n4. 극단적 종횡비 (AR > {df['avg_aspect_ratio'].quantile(0.9):.1f})")
    print(f"   - 특이한 텍스트 배치 (세로 긴 혹은 가로 긴)")
    print(f"   - 앵커 박스 매칭 문제 가능")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
