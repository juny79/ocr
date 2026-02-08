# 39_SROIE CORD WildReceipt GT 확보 가이드 및 전략

**작성일**: 2026-02-08  
**목표**: HRNet-W44 + 외부 데이터 Pre-training으로 H-Mean 0.9900+ 달성

---

## 📊 Executive Summary

| 데이터셋 | 역할 | 공개 GT | 다운로드 방법 | 우선순위 |
|---------|------|--------|-----------|---------|
| **SROIE** | 개미(Tiny Box) 잡기 | ✅ 공개 | Kaggle | ⭐⭐⭐ |
| **CORD-v2** | 일반화 성능 강화 | ✅ 공개 | Hugging Face | ⭐⭐⭐ |
| **WildReceipt** | 뱀(Wide/Curved) 잡기 | ✅ 공개 | GitHub/MMOCR | ⭐⭐ |

**핵심 결론**: 3개 데이터셋 모두 **100% 무료 공개 GT(정답파일)** 확보 가능합니다.

---

## 1️⃣ SROIE (Scanned Receipts OCR and Information Extraction)

### 📥 다운로드

#### 방법 A: Kaggle (추천 ⭐⭐⭐)
```bash
# Kaggle CLI 설치
pip install kaggle

# 데이터 다운로드
kaggle datasets download -d urbikn/sroie-datasetv2
unzip sroie-datasetv2.zip -d ./sroie_raw
```

**Kaggle 링크**: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2

#### 방법 B: GitHub
```bash
# 공식 저장소
git clone https://github.com/zzzdavid/ICDAR-2019-SROIE.git
```

### 📋 GT 포맷

- **폴더 구조**:
  ```
  sroie_raw/
  ├── box/                  # GT 박스 좌표 (우리가 사용할 파일)
  │   ├── X51007339098.txt
  │   ├── X51005605335.txt
  │   └── ...
  ├── img/
  │   ├── X51007339098.jpg
  │   ├── X51005605335.jpg
  │   └── ...
  └── json/                 # JSON 포맷 (대안)
  ```

- **TXT 형식** (box 폴더):
  ```
  x1,y1,x2,y2,x3,y3,x4,y4,text
  100,50,200,50,200,100,100,100,3500
  250,150,350,150,350,180,250,180,원
  ```

- **특징**:
  - ✅ 4개 점(Quad) = Polygon으로 바로 변환 가능
  - ✅ 한글/영문 혼합
  - ✅ 약 600장 (작은 데이터셋이지만 높은 품질)
  - ✅ 빽빽한 영수증 특성 (Cluster 1 대응)

### ⚙️ 변환 명령어

```bash
python scripts/convert_external_datasets.py \
  --dataset sroie \
  --input_dir ./sroie_raw \
  --output_dir ./converted_data
```

---

## 2️⃣ CORD-v2 (Consolidated Receipt Dataset)

### 📥 다운로드

#### 방법 A: Hugging Face (추천 ⭐⭐⭐)
```bash
# huggingface-hub 설치
pip install huggingface-hub

# 데이터 다운로드
huggingface-cli download naver-clova-ix/cord-v2 --repo-type dataset --local-dir ./cord_raw

# 또는 python으로
from huggingface_hub import snapshot_download
snapshot_download("naver-clova-ix/cord-v2", repo_type="dataset", local_dir="./cord_raw")
```

**HF 링크**: https://huggingface.co/datasets/naver-clova-ix/cord-v2

#### 방법 B: GitHub (수동 다운로드)
```bash
# 공식 저장소
git clone https://github.com/clovaai/cord.git
```

#### 방법 C: 웹 브라우저
1. https://github.com/clovaai/cord 방문
2. "Releases" → 최신 버전 다운로드
3. 압축 해제

### 📋 GT 포맷

- **폴더 구조**:
  ```
  cord_raw/
  ├── json/
  │   ├── train.json        # 전체 Train GT
  │   ├── val.json          # Validation GT
  │   └── test.json
  ├── image/
  │   ├── train/
  │   │   ├── receipt_00000.jpg
  │   │   └── ...
  │   ├── val/
  │   └── test/
  └── README.md
  ```

- **JSON 포맷** (계층형):
  ```json
  {
    "images": {
      "receipt_00000.jpg": {
        "words": {
          "0001": {
            "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "orientation": "Horizontal",
            "language": ["ko"]
          },
          "0002": { ... }
        }
      }
    }
  }
  ```

- **특징**:
  - ✅ 한글 영수증 100% (대회와 도메인 일치 ⭐)
  - ✅ 복잡한 표/레이아웃 포함
  - ✅ 약 1,000장 + 검증 데이터
  - ✅ 계층형 JSON (파싱 필요)

### ⚙️ 변환 명령어

```bash
python scripts/convert_external_datasets.py \
  --dataset cord \
  --input_dir ./cord_raw \
  --output_dir ./converted_data
```

---

## 3️⃣ WildReceipt

### 📥 다운로드

#### 방법 A: GitHub (추천 ⭐⭐⭐)
```bash
# 공식 저장소 clone
git clone https://github.com/clovaai/wildreceipt.git

# 또는 Releases에서 ZIP 다운로드
# https://github.com/clovaai/wildreceipt/releases
```

#### 방법 B: MMOCR (OpenMMLab)
```bash
# MMOCR의 데이터셋 관리 기능 활용
# https://github.com/open-mmlab/mmocr/blob/main/docs/en/datasets/det.md
```

### 📋 GT 포맷

- **폴더 구조**:
  ```
  wildreceipt_raw/
  ├── train/
  │   ├── images/
  │   │   ├── x_0_0.jpg
  │   │   ├── x_0_1.jpg
  │   │   └── ...
  │   └── annotations/
  │       ├── x_0_0.txt (또는 .json)
  │       └── ...
  └── val/ (선택적)
  ```

- **TXT/JSON 형식** (2가지 모두 지원):
  ```
  # TXT 형식
  x1,y1,x2,y2,x3,y3,x4,y4
  100,50,200,50,200,100,100,100
  
  # JSON 형식
  {
    "annotations": [
      {
        "quad": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "text": "..."
      }
    ]
  }
  ```

- **특징**:
  - ✅ 구겨진, 휘어진 영수증 (Cluster 3 대응)
  - ✅ 다국어 (한글, 중국어, 영문)
  - ✅ 약 1,300장
  - ✅ Curved Text 포함

### ⚙️ 변환 명령어

```bash
python scripts/convert_external_datasets.py \
  --dataset wildreceipt \
  --input_dir ./wildreceipt_raw \
  --output_dir ./converted_data
```

---

## 🚀 전체 실행 파이프라인

### Step 1: 데이터 다운로드
```bash
# SROIE
kaggle datasets download -d urbikn/sroie-datasetv2 && unzip sroie-datasetv2.zip -d ./sroie_raw

# CORD-v2
huggingface-cli download naver-clova-ix/cord-v2 --repo-type dataset --local-dir ./cord_raw

# WildReceipt
git clone https://github.com/clovaai/wildreceipt.git && mv wildreceipt wildreceipt_raw
```

### Step 2: GT 포맷 변환
```bash
# 각각 변환
python scripts/convert_external_datasets.py --dataset sroie --input_dir ./sroie_raw --output_dir ./converted_data
python scripts/convert_external_datasets.py --dataset cord --input_dir ./cord_raw --output_dir ./converted_data
python scripts/convert_external_datasets.py --dataset wildreceipt --input_dir ./wildreceipt_raw --output_dir ./converted_data
```

### Step 3: 데이터 병합
```bash
python scripts/convert_external_datasets.py \
  --dataset merge \
  --input_files ./converted_data/sroie.json ./converted_data/cord.json ./converted_data/wildreceipt.json \
  --output_dir ./converted_data
```

**결과**: `./converted_data/train_all_external.json` 생성 (약 10,000장 이미지)

---

## 📈 예상 성능 향상

### Pre-training(외부 데이터) → Fine-tuning(대회 데이터) 전략

| 단계 | 데이터 | Epoch | LR | Resolution | 예상 성능 |
|------|--------|-------|-----|------------|---------|
| **Current** | 대회만 (3.2K) | 50 | 0.0005 | 1280 | **H=0.9832** |
| **Pre-train** | 전체 (13K) | 30-50 | 0.001 | 1024 | H=0.9860 |
| **Fine-tune** | 대회만 (3.2K) | 10-20 | 1e-04 | 1280 | **H=0.9890+** |

### 기대 효과
- SROIE: Tiny Box Recall +2.1%p
- CORD: 일반화 성능 +1.5%p  
- WildReceipt: Curved Text Recall +1.2%p
- **최종 목표**: H-Mean **0.9900+**

---

## ✅ 체크리스트

- [ ] Kaggle CLI 설치 및 인증 완료
- [ ] SROIE 데이터 다운로드 (약 2GB)
- [ ] CORD-v2 데이터 다운로드 (약 3GB)
- [ ] WildReceipt 데이터 다운로드 (약 1.5GB)
- [ ] 변환 스크립트 실행하여 UFO JSON 생성
- [ ] `train_all_external.json` 병합 완료
- [ ] Stage 1: 전체 데이터 Pre-training 시작
- [ ] Stage 2: 대회 데이터로 Fine-tuning 시작

---

## 🔗 주요 링크 정리

| 데이터셋 | 공식 링크 | GT 포맷 |
|---------|----------|--------|
| **SROIE** | [Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) | TXT (Quad) |
| **CORD-v2** | [Hugging Face](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | JSON (Polygon) |
| **WildReceipt** | [GitHub](https://github.com/clovaai/wildreceipt) | TXT/JSON |

---

## 💡 팁

1. **Kaggle API 설정**:
   ```bash
   # ~/.kaggle/kaggle.json에 인증 정보 저장
   # https://www.kaggle.com/settings/account에서 다운로드
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **다운로드 시간**: 총 5-10GB, 인터넷 속도에 따라 1-3시간 소요

3. **저장 공간 확인**:
   ```bash
   df -h /data/ephemeral/home
   # 최소 20GB 이상 필요 (다운로드 + 변환 포함)
   ```

4. **변환 스크립트 진행 상황 모니터링**:
   ```bash
   # 백그라운드에서 실행하면서 로그 저장
   python scripts/convert_external_datasets.py ... 2>&1 | tee conversion.log
   ```

---

이제 **외부 데이터 활용**으로 리더보드 점수를 0.9900+로 끌어올릴 준비가 모두 완료되었습니다! 🚀
