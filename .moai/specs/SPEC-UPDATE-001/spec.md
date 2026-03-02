---
id: SPEC-UPDATE-001
version: "1.0.0"
status: draft
created: "2026-03-02"
updated: "2026-03-02"
author: "허태명"
priority: high
---

## HISTORY

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-03-02 | 허태명 | Initial SPEC creation |

---

# SPEC-UPDATE-001: 2019 Kaggle 프로젝트 현대화 (uv + Jupyter)

## 1. Overview

2019년 Kaggle Korea House Price Prediction 대회 솔루션 코드를 현재 로컬 환경(macOS, Python 3.14, uv)에서 실행 가능하도록 현대화한다.

## 2. EARS Requirements

### 2.1 Ubiquitous Requirements

- REQ-U-001: 프로젝트는 `pyproject.toml`을 통해 모든 의존성을 관리해야 한다.
- REQ-U-002: `uv` 가상 환경에서 모든 노트북이 실행 가능해야 한다.
- REQ-U-003: 기존 ML 파이프라인의 로직과 결과를 보존해야 한다.

### 2.2 Event-Driven Requirements

- REQ-E-001: WHEN `uv sync`를 실행하면 THEN 모든 의존성이 설치되어야 한다.
- REQ-E-002: WHEN Jupyter 노트북을 실행하면 THEN uv 가상 환경의 커널로 실행되어야 한다.
- REQ-E-003: WHEN `utils.py`를 import 하면 THEN 모든 deprecated API가 현대화된 버전으로 동작해야 한다.

### 2.3 State-Driven Requirements

- REQ-S-001: WHILE Python 3.14 환경에서 WHEN 라이브러리 호환성 문제가 발생하면 THEN Python 3.12로 폴백한다.
- REQ-S-002: WHILE 노트북 실행 중 WHEN pre-computed CSV 데이터가 존재하면 THEN 중간 계산 단계를 건너뛸 수 있어야 한다.

### 2.4 Unwanted Behavior Requirements

- REQ-UW-001: 기존 ML 모델의 하이퍼파라미터를 변경해서는 안 된다.
- REQ-UW-002: 기존 데이터 전처리 로직을 변경해서는 안 된다.
- REQ-UW-003: 노트북 출력 결과(시각화, CV 스코어)가 기존과 크게 달라져서는 안 된다.

### 2.5 Optional Requirements

- REQ-O-001: WHERE 가능하면 `ruff`를 린터/포매터로 설정한다.

## 3. Technical Approach

### 3.1 Phase 1: 프로젝트 환경 설정

1. `pyproject.toml` 생성
   - Python version: `>=3.12,<3.15` (3.14 우선, 호환 문제 시 3.12 폴백)
   - 의존성 목록 정의 (아래 참조)
   - ruff 설정 포함

2. `.python-version` 생성
   - uv가 사용할 Python 버전 지정

3. `.gitignore` 업데이트
   - `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` 추가

### 3.2 Phase 2: 의존성 정의

핵심 의존성 목록:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
tensorflow>=2.15
scipy>=1.11
psutil>=5.9
tqdm>=4.65
ipykernel>=6.25
jupyter>=1.0
```

### 3.3 Phase 3: 코드 현대화 (utils.py)

수정 대상 및 변경 내역:

| Line | 현재 코드 | 변경 코드 | 이유 |
|------|----------|----------|------|
| 30 | `from sklearn.preprocessing import Imputer` | `from sklearn.impute import SimpleImputer` | sklearn 1.0에서 제거됨 |
| 43-48 | `from keras.models import ...` | `from tensorflow.keras.models import ...` | Standalone keras 미지원 |
| 49 | `from keras.utils import plot_model` | `from tensorflow.keras.utils import plot_model` | keras → tf.keras |
| 52 | `from tensorflow import set_random_seed` | 제거 후 `tf.random.set_seed()` 사용 | TF2 API |
| 60 | `from tqdm import tqdm_notebook as tqdm` | `from tqdm.notebook import tqdm` | tqdm API 변경 |
| 69-73 | `tf.ConfigProto`, `tf.Session`, `K.set_session` | `tf.random.set_seed(RANDOM_SEED)` | TF2 eager execution |

### 3.4 Phase 4: 노트북 현대화

1. **geo-data-eda-and-feature-engineering.ipynb**
   - `sns.distplot()` → `sns.histplot(kde=True)` 또는 `sns.kdeplot()`

2. **Generate Neighbor Info.ipynb**
   - `from utils import *` 경로 확인
   - `tqdm_notebook` 호환 확인

3. **Generate Neighbor Stat.ipynb**
   - `from utils import *` 경로 확인

4. **Stacking Ensemble.ipynb**
   - `normalize=True` 파라미터 제거 (Ridge, Lasso, ElasticNet)
   - `from utils import *` 경로 호환 확인

### 3.5 Phase 5: Jupyter 커널 등록

```bash
uv sync
uv run python -m ipykernel install --user --name kaggle-2019 --display-name "Kaggle 2019 (uv)"
```

## 4. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Python 3.14에서 TensorFlow 미지원 | 높음 | `.python-version`에 3.12 지정, uv가 자동 설치 |
| sklearn normalize 제거로 모델 결과 변경 | 중간 | normalize는 전처리 단계이므로 결과에 미미한 영향 |
| seaborn API 변경으로 시각화 차이 | 낮음 | histplot이 distplot과 유사한 출력 제공 |
| Keras 3.x vs tf.keras 호환성 | 중간 | tensorflow>=2.15 사용으로 tf.keras 안정성 확보 |

## 5. Scope

### In Scope
- pyproject.toml + uv 환경 설정
- Deprecated API 현대화 (utils.py, notebooks)
- Jupyter 커널 등록
- .gitignore 업데이트

### Out of Scope
- ML 모델 성능 개선
- 새로운 feature engineering 추가
- 코드 리팩토링 (동작 변경 없는 최소 수정만)
- CI/CD 파이프라인 설정
