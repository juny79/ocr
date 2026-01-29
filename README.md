# Receipt Text Detection Project

## 📋 프로젝트 개요

DBNet 기반의 영수증 텍스트 감지(Text Detection) 프로젝트입니다.

## 📊 현재 성능 (10 Epochs)

- **H-Mean**: 0.8818
- **Precision**: 0.9651
- **Recall**: 0.8194

## 🏗️ 프로젝트 구조

```
.
├── baseline_code/          # 베이스라인 코드
│   ├── configs/           # 설정 파일
│   ├── ocr/              # 핵심 모듈
│   ├── runners/          # 실행 스크립트
│   └── requirements.txt  # 의존성 패키지
├── data/                 # 데이터셋
└── baseline_analysis_report.md  # 성능 분석 보고서

```

## 🚀 시작하기

### 환경 설정

```bash
pip install -r baseline_code/requirements.txt
```

### 데이터셋 경로 설정

`baseline_code/configs/preset/datasets/db.yaml` 파일에서 데이터셋 경로를 수정하세요:

```yaml
dataset_base_path: "/data/datasets/"
```

### 학습

```bash
python baseline_code/runners/train.py preset=example
```

### 테스트

```bash
python baseline_code/runners/test.py preset=example "checkpoint_path='{checkpoint_path}'"
```

### 예측

```bash
python baseline_code/runners/predict.py preset=example "checkpoint_path='{checkpoint_path}'"
```

## 📚 기술 스택

- **Framework**: PyTorch Lightning
- **Model**: DBNet (Differentiable Binarization)
- **Backbone**: ResNet18 (timm)
- **Config**: Hydra
- **Augmentation**: Albumentations

## 📈 개선 방향

상세한 분석 내용은 [baseline_analysis_report.md](baseline_analysis_report.md)를 참조하세요.

### 우선순위 개선 항목

1. **Postprocessing 임계값 조정** (`box_thresh: 0.4 → 0.3`)
2. **Data Augmentation 추가** (Rotation, Brightness, Scale)
3. **Learning Rate Scheduler 수정** (CosineAnnealing)
4. **학습 Epochs 증가** (10 → 50+)

## 📝 참고 자료

- [DBNet Paper](https://arxiv.org/pdf/1911.08947.pdf)
- [CLEval Metric](https://github.com/clovaai/CLEval)

## 📄 License

이 프로젝트는 교육 목적으로 제작되었습니다.
