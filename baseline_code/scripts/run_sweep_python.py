#!/usr/bin/env python3
"""
WandB Sweep - Python API 방식
Non-interactive 환경에서 안전하게 동작
"""

import os
import sys
import yaml
import wandb
from pathlib import Path

def main():
    print("=" * 50)
    print("WandB Sweep - Learning Rate 최적화")
    print("=" * 50)
    print()
    
    # WANDB_API_KEY 확인
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("❌ WANDB_API_KEY 환경변수가 설정되지 않았습니다.")
        print()
        print("다음 명령어로 API Key를 설정하세요:")
        print("export WANDB_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("✅ WANDB_API_KEY 확인됨")
    print()
    
    # Sweep 설정 파일 로드
    sweep_config_path = Path(__file__).parent.parent / "configs" / "sweep_efficientnet_b4_lr_optimized.yaml"
    
    if not sweep_config_path.exists():
        print(f"❌ Sweep 설정 파일을 찾을 수 없습니다: {sweep_config_path}")
        sys.exit(1)
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    print("📋 Sweep 정보")
    print("-" * 50)
    print("Base 성능: 96.37% (Postprocessing 최적화 완료)")
    print("목표: 96.50%+")
    print("전략: Learning Rate + Weight Decay 최적화")
    print()
    print(f"Method: {sweep_config['method']}")
    print(f"Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print()
    
    # Sweep 초기화 (Python API 사용)
    try:
        print("🚀 Sweep 초기화 중...")
        # API key 환경변수 설정 (login 대신)
        os.environ["WANDB_API_KEY"] = api_key
        
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project="ocr-efficientnet-b4-lr-optimization"
        )
        
        print(f"✅ Sweep ID: {sweep_id}")
        print()
        
        # Sweep agent 실행 함수
        def train():
            """WandB Sweep이 호출할 학습 함수"""
            run = wandb.init()
            
            # Sweep에서 제안한 하이퍼파라미터 가져오기
            config = wandb.config
            
            print("=" * 50)
            print(f"Run #{run.name}")
            print("=" * 50)
            print(f"LR: {config.lr}")
            print(f"Weight Decay: {config.weight_decay}")
            print(f"T_Max: {config.t_max}")
            print(f"eta_min: {config.eta_min}")
            print()
            
            # 학습 실행
            import subprocess
            cmd = [
                "python", "runners/train.py",
                f"preset=efficientnet_b4_lr_optimized",
                f"exp_name=sweep_{run.name}",
                f"models.optimizer.lr={config.lr}",
                f"models.optimizer.weight_decay={config.weight_decay}",
                f"models.lr_scheduler.t_max={config.t_max}",
                f"models.lr_scheduler.eta_min={config.eta_min}",
                "trainer.max_epochs=22"
            ]
            
            print("실행 명령:")
            print(" ".join(cmd))
            print()
            
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode != 0:
                print(f"❌ 학습 실패: Return code {result.returncode}")
                wandb.finish(exit_code=1)
            else:
                print("✅ 학습 완료")
                wandb.finish(exit_code=0)
        
        # Sweep 실행
        num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
        
        print("=" * 50)
        print(f"WandB Sweep 실행 중 ({num_runs}회)...")
        print("=" * 50)
        print()
        print("진행상황은 WandB 대시보드에서 확인:")
        print(f"https://wandb.ai/your-username/ocr-efficientnet-b4-lr-optimization/sweeps/{sweep_id}")
        print()
        
        # Agent 실행
        wandb.agent(sweep_id, function=train, count=num_runs)
        
        print()
        print("=" * 50)
        print("✅ Sweep 완료!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Sweep 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
