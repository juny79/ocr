"""
그리드 서치 결과 시각화 및 분석
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_file):
    """결과 파일 로드"""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_heatmap(results, save_path=None):
    """H-Mean 히트맵 생성"""
    experiments = [e for e in results['experiments'] if e['metrics']['success']]
    
    if not experiments:
        print("유효한 실험 결과가 없습니다.")
        return
    
    # Thresh와 Box Thresh 범위 추출
    threshs = sorted(list(set([e['thresh'] for e in experiments])))
    box_threshs = sorted(list(set([e['box_thresh'] for e in experiments])))
    
    # 히트맵 데이터 생성
    heatmap_data = np.zeros((len(threshs), len(box_threshs)))
    
    for i, thresh in enumerate(threshs):
        for j, box_thresh in enumerate(box_threshs):
            for exp in experiments:
                if abs(exp['thresh'] - thresh) < 0.001 and abs(exp['box_thresh'] - box_thresh) < 0.001:
                    heatmap_data[i, j] = exp['metrics']['hmean']
                    break
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{bt:.3f}" for bt in box_threshs],
        yticklabels=[f"{t:.3f}" for t in threshs],
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'H-Mean'}
    )
    plt.xlabel('Box Threshold', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title('후처리 파라미터 그리드 서치 결과 (H-Mean)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_precision_recall_plot(results, save_path=None):
    """Precision-Recall 산점도"""
    experiments = [e for e in results['experiments'] if e['metrics']['success']]
    
    if not experiments:
        return
    
    precisions = [e['metrics']['precision'] for e in experiments]
    recalls = [e['metrics']['recall'] for e in experiments]
    hmeans = [e['metrics']['hmean'] for e in experiments]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        recalls, precisions,
        c=hmeans,
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    plt.colorbar(scatter, label='H-Mean')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall 분포 (색상: H-Mean)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_parameter_sensitivity_plot(results, save_path=None):
    """파라미터 민감도 분석"""
    experiments = [e for e in results['experiments'] if e['metrics']['success']]
    
    if not experiments:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Thresh 영향
    thresh_groups = {}
    for exp in experiments:
        thresh = exp['thresh']
        if thresh not in thresh_groups:
            thresh_groups[thresh] = []
        thresh_groups[thresh].append(exp['metrics']['hmean'])
    
    threshs = sorted(thresh_groups.keys())
    means = [np.mean(thresh_groups[t]) for t in threshs]
    stds = [np.std(thresh_groups[t]) for t in threshs]
    
    ax1.errorbar(threshs, means, yerr=stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('H-Mean (평균 ± 표준편차)', fontsize=12)
    ax1.set_title('Threshold의 H-Mean 영향', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Box Thresh 영향
    box_thresh_groups = {}
    for exp in experiments:
        box_thresh = exp['box_thresh']
        if box_thresh not in box_thresh_groups:
            box_thresh_groups[box_thresh] = []
        box_thresh_groups[box_thresh].append(exp['metrics']['hmean'])
    
    box_threshs = sorted(box_thresh_groups.keys())
    means = [np.mean(box_thresh_groups[bt]) for bt in box_threshs]
    stds = [np.std(box_thresh_groups[bt]) for bt in box_threshs]
    
    ax2.errorbar(box_threshs, means, yerr=stds, fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax2.set_xlabel('Box Threshold', fontsize=12)
    ax2.set_ylabel('H-Mean (평균 ± 표준편차)', fontsize=12)
    ax2.set_title('Box Threshold의 H-Mean 영향', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"민감도 분석 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_analysis(results):
    """결과 분석 출력"""
    experiments = [e for e in results['experiments'] if e['metrics']['success']]
    
    if not experiments:
        print("유효한 실험 결과가 없습니다.")
        return
    
    print("\n" + "="*80)
    print("그리드 서치 결과 분석")
    print("="*80)
    
    # 기본 통계
    hmeans = [e['metrics']['hmean'] for e in experiments]
    print(f"\nH-Mean 통계:")
    print(f"  최대: {max(hmeans):.6f}")
    print(f"  최소: {min(hmeans):.6f}")
    print(f"  평균: {np.mean(hmeans):.6f}")
    print(f"  표준편차: {np.std(hmeans):.6f}")
    
    # 최고 결과
    best = results.get('best')
    if best:
        print(f"\n🏆 최고 성능:")
        print(f"  Thresh: {best['thresh']:.4f}")
        print(f"  Box Thresh: {best['box_thresh']:.4f}")
        print(f"  H-Mean: {best['hmean']:.6f}")
        print(f"  Precision: {best['precision']:.6f}")
        print(f"  Recall: {best['recall']:.6f}")
    
    # Baseline과 비교
    baseline = results.get('baseline', {})
    if baseline and best:
        baseline_sub = baseline.get('submission_score', {})
        print(f"\n📊 Baseline 대비:")
        print(f"  Baseline Submission H-Mean: {baseline_sub.get('hmean', 0):.4f}")
        print(f"  Best Validation H-Mean: {best['hmean']:.6f}")
    
    # 상위 10개
    sorted_experiments = sorted(experiments, key=lambda x: x['metrics']['hmean'], reverse=True)[:10]
    print(f"\n📈 상위 10개 파라미터 조합:")
    for i, exp in enumerate(sorted_experiments, 1):
        print(f"  {i}. thresh={exp['thresh']:.4f}, box_thresh={exp['box_thresh']:.4f} "
              f"→ H-Mean: {exp['metrics']['hmean']:.6f} "
              f"(P: {exp['metrics']['precision']:.4f}, R: {exp['metrics']['recall']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='그리드 서치 결과 시각화')
    parser.add_argument('results_file', type=str, help='결과 JSON 파일 경로')
    parser.add_argument('--output-dir', type=str, default='grid_search_results',
                       help='그래프 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 결과 로드
    results = load_results(args.results_file)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 파일명 base
    base_name = Path(args.results_file).stem
    
    # 분석 출력
    print_analysis(results)
    
    # 히트맵
    heatmap_path = output_dir / f"{base_name}_heatmap.png"
    create_heatmap(results, heatmap_path)
    
    # Precision-Recall 그래프
    pr_path = output_dir / f"{base_name}_precision_recall.png"
    create_precision_recall_plot(results, pr_path)
    
    # 민감도 분석
    sensitivity_path = output_dir / f"{base_name}_sensitivity.png"
    create_parameter_sensitivity_plot(results, sensitivity_path)
    
    print(f"\n모든 그래프가 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
