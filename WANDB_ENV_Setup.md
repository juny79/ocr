# 환경변수 설정 가이드

## 🔐 WANDB_API_KEY 설정 방법

### 1️⃣ **API Key 확인 (필수 사전 작업)**

1. **W&B 웹사이트**: https://wandb.ai/settings/keys
2. **로그인** 후 API Key 복사
3. 다음 단계에서 사용

---

## ✅ **권장 방법: .env 파일 (개발 환경)**

### **장점**
- ✅ 간단하고 직관적
- ✅ 로컬 개발에 최적화
- ✅ API Key가 git에 커밋되지 않음
- ✅ 팀원과 쉽게 공유 가능

### **설정 방법**

#### **Step 1: .env 파일 생성**

`.env` 파일을 프로젝트 루트에 생성:

```bash
# 프로젝트 루트 디렉토리에서
cat > .env << EOF
WANDB_API_KEY=your-actual-api-key-here
WANDB_PROJECT=ocr-receipt-detection
WANDB_ENTITY=your-username
WANDB_MODE=online
EOF
```

#### **Step 2: python-dotenv 설치**

```bash
pip install python-dotenv
```

또는 requirements.txt에 추가:
```
python-dotenv==1.2.1
```

#### **Step 3: 확인**

```bash
# .env 파일이 생성되었는지 확인
ls -la .env

# 내용 확인
cat .env
```

#### **Step 4: 스크립트 실행**

```bash
# 자동으로 .env 파일에서 환경변수 로드됨
python runners/train.py preset=example wandb=True
```

---

## 🖥️ **대체 방법: .bashrc (서버 환경)**

### **장점**
- ✅ 항상 활성화 (재로그인 필요 없음)
- ✅ 모든 터미널 세션에서 적용
- ✅ 서버 환경에 적합

### **설정 방법**

#### **Step 1: .bashrc 수정**

```bash
# 터미널에서 직접 입력
nano ~/.bashrc

# 또는 명령어로 추가
echo 'export WANDB_API_KEY="your-api-key-here"' >> ~/.bashrc
```

#### **Step 2: 설정 적용**

```bash
source ~/.bashrc
```

#### **Step 3: 확인**

```bash
echo $WANDB_API_KEY
# 출력: your-api-key-here
```

---

## 🚀 **임시 방법: 런타임 설정**

한 번만 실행할 때:

```bash
# 방법 1: 별도 export
export WANDB_API_KEY="your-api-key-here"
python runners/train.py preset=example wandb=True

# 방법 2: 한 줄로 (권장)
WANDB_API_KEY="your-api-key-here" python runners/train.py preset=example wandb=True
```

---

## 🐳 **Docker 환경**

### **Build Time 설정**

```bash
docker build \
  --build-arg WANDB_API_KEY="your-api-key-here" \
  -t ocr-text-detection:latest .
```

### **Runtime 설정**

```bash
docker run \
  -e WANDB_API_KEY="your-api-key-here" \
  ocr-text-detection:latest
```

### **docker-compose.yml**

```yaml
version: '3.8'

services:
  training:
    build: .
    environment:
      WANDB_API_KEY: ${WANDB_API_KEY}
      WANDB_PROJECT: ocr-receipt-detection
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
```

실행:
```bash
WANDB_API_KEY="your-api-key-here" docker-compose up
```

---

## 🔒 **보안 체크리스트**

| 항목 | 상태 | 확인 |
|------|------|------|
| API Key가 .gitignore에 제외됨 | ✅ | `git check-ignore .env` |
| .env 파일이 git 추적 안 됨 | ✅ | `git status` |
| public 저장소에서 key 노출 안 함 | ✅ | GitHub 확인 |
| 강력한 권한 설정 | ✅ | `chmod 600 .env` |

---

## 🧪 **환경변수 테스트**

### **Python에서 확인**

```python
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 확인
wandb_key = os.getenv('WANDB_API_KEY')
print(f"WANDB_API_KEY: {wandb_key[:10]}..." if wandb_key else "Not set")
```

### **Bash에서 확인**

```bash
# 환경변수 확인
echo $WANDB_API_KEY

# 또는 env 명령어
env | grep WANDB
```

---

## 📊 **설정 우선순위**

1. **런타임 환경변수** (가장 높음)
   ```bash
   WANDB_API_KEY="key" python script.py
   ```

2. **.env 파일** (python-dotenv 로드 시)
   ```
   WANDB_API_KEY=key
   ```

3. **.bashrc / 환경 변수**
   ```bash
   export WANDB_API_KEY=key
   ```

4. **기본값** (없으면 인터랙티브 로그인)

---

## 🆘 **트러블슈팅**

### **문제: "WANDB_API_KEY not found"**

해결책:
```bash
# 1. .env 파일 확인
ls -la .env

# 2. 내용 확인
cat .env

# 3. python-dotenv 설치 확인
pip show python-dotenv

# 4. train.py에 load_dotenv() 호출 확인
grep -n "load_dotenv" runners/train.py
```

### **문제: "Permission denied" (API Key 사용 불가)**

해결책:
```bash
# API Key 재생성
# https://wandb.ai/settings/keys

# 권한 설정
chmod 600 .env
```

### **문제: 여러 프로젝트에서 충돌**

해결책:
```bash
# 프로젝트별 .env 파일 생성
# project1/.env
# project2/.env

# 또는 환경변수로 override
WANDB_PROJECT=other-project python script.py
```

---

## 📝 **각 방법 비교표**

| 방법 | 개발 | 서버 | CI/CD | 보안 | 간편성 |
|------|------|------|-------|------|--------|
| **.env 파일** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **.bashrc** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **런타임** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Docker** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## ✅ **다음 단계**

1. **API Key 복사**: https://wandb.ai/settings/keys
2. **.env 파일 생성** 또는 **.bashrc 수정**
3. **학습 실행**: `python runners/train.py preset=example wandb=True`
4. **W&B Dashboard 확인**: https://wandb.ai/

---

**현재 프로젝트 상태:**
- ✅ .env 파일 생성됨
- ✅ python-dotenv 설치됨
- ✅ train.py에 load_dotenv() 통합됨
- ✅ .env가 .gitignore에 등록됨
