# 2019 2nd ML month with KaKR - House Price Prediction

2019 Kaggle Korea House Price Prediction 대회 11위 솔루션

- **Kaggle Kernel**: https://www.kaggle.com/tmheo74/11th-solution-public-98316-private-99336

## 환경 설정

### 필수 요건

- [uv](https://docs.astral.sh/uv/) (Python 패키지 매니저)

### 설치 및 실행

```bash
# 의존성 설치 (Python 3.12 자동 설치 포함)
uv sync

# Jupyter 커널 등록
uv run python -m ipykernel install --user --name kaggle-2019 --display-name "Kaggle 2019 (uv)"

# Jupyter 노트북 실행
uv run jupyter notebook
```

## 프로젝트 구조

```
├── input/                  # 대회 데이터 및 중간 결과물
│   ├── train.csv           # 학습 데이터
│   ├── test.csv            # 테스트 데이터
│   └── *.csv               # Pre-computed 중간 결과물
├── notebook/
│   ├── utils.py            # 공유 ML 유틸리티 (모델 래퍼, OOF, 스태킹)
│   ├── geo-data-eda-and-feature-engineering.ipynb  # EDA 및 지리 피처 엔지니어링
│   ├── Generate Neighbor Info.ipynb                # Haversine 거리 이웃 계산
│   ├── Generate Neighbor Stat.ipynb                # 이웃 통계 집계
│   └── Stacking Ensemble.ipynb                     # 멀티레벨 스태킹 앙상블
├── pyproject.toml          # 프로젝트 설정 및 의존성
├── .python-version         # Python 3.12 (uv용)
└── uv.lock                 # 의존성 잠금 파일
```

## 노트북 실행 순서

1. `geo-data-eda-and-feature-engineering.ipynb` - EDA (독립 실행 가능)
2. `Generate Neighbor Info.ipynb` → `neighbor_info.csv` 생성
3. `Generate Neighbor Stat.ipynb` → `neighbor_*_stat.csv` 생성
4. `Stacking Ensemble.ipynb` → 최종 앙상블 모델

> 2\~3단계의 중간 결과물이 `input/`에 이미 포함되어 있으므로, 4번 노트북을 바로 실행할 수 있습니다.

## 기술 스택

- **Python**: 3.12
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow/Keras
- **데이터**: pandas, numpy, scipy
- **시각화**: matplotlib, seaborn
- **패키지 관리**: uv