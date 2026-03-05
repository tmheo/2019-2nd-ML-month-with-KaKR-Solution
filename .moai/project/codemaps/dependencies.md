# 의존성 그래프

## 외부 패키지 의존성

### 데이터 처리 라이브러리

**pandas** (0.23+)
- 용도: 데이터프레임 처리 (load_data, 피처 병합, 통계 집계)
- 주요 함수: read_csv, concat, merge, groupby, get_dummies, apply
- 호출 위치: utils.py 전반, 모든 노트북

**numpy** (1.14+)
- 용도: 수치 배열 연산 (거리 계산, 변환, 통계)
- 주요 함수: array, concatenate, zeros, unique, isinf, radians, degrees, arctan2, sqrt
- 호출 위치: haversine_array, bearing_array, stacking, 모든 래퍼

**scipy** (1.0+)
- 용도: 통계 및 과학 계산
- 모듈:
  - scipy.stats: skew (분포 왜도), norm, boxcox1p (분포 정정)
  - scipy.special: boxcox1p
  - scipy.cluster.hierarchy: 계층적 클러스터링

### 머신러닝 라이브러리

**scikit-learn** (0.19+)
- 용도: 데이터 전처리, 모델 학습, 교차 검증
- 주요 클래스:
  - preprocessing: LabelEncoder, StandardScaler, RobustScaler, Imputer
  - model_selection: train_test_split, KFold, StratifiedKFold, cross_val_score
  - metrics: mean_squared_error
  - ensemble: GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
  - linear_model: ElasticNet, Lasso, Ridge
  - svm: SVR
  - decomposition: PCA (좌표 변환)
  - cluster: KMeans (지리적 클러스터링)
- 호출 위치: load_data, SklearnWrapper, 모든 파이프라인

**XGBoost** (0.70+)
- 용도: 그래디언트 부스팅 회귀 모델
- 주요 클래스: xgb.train, xgb.DMatrix
- 호출 위치: XgbWrapper.train, XgbWrapper.predict
- 설정: xgboost 파라미터 dict (learning_rate, max_depth, etc)

**LightGBM** (2.1+)
- 용도: 빠른 그래디언트 부스팅 모델
- 주요 클래스: lgb.train, lgb.Dataset, lgb.plot_importance
- 호출 위치: LgbmWrapper.train, LgbmWrapper.predict, LgbmWrapper.plot_importance
- 특징: 속도 최적화, 범주형 변수 지원

**CatBoost** (0.8+)
- 용도: 범주형 변수 최적화 부스팅
- 주요 클래스: cat.CatBoost, cat.Pool
- 호출 위치: CatWrapper.train, CatWrapper.predict
- 특징: 자동 범주형 처리, cat_features 파라미터

### 딥러닝 라이브러리

**Keras** (2.1+)
- 용도: 신경망 모델 구축 및 학습
- 주요 클래스:
  - models: Sequential, Model
  - layers: Dense, Embedding, Reshape, Concatenate, Input, Flatten, BatchNormalization, Dropout
  - optimizers: Adam, SGD, RMSprop
  - callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  - utils: plot_model
  - backend: K (Keras 백엔드, 세션 관리)
- 호출 위치: KerasWrapper, KerasEmbeddingWrapper

**TensorFlow** (1.5+)
- 용도: Keras 백엔드, 분산 학습
- 주요 함수:
  - tf.Session, tf.get_default_graph, tf.ConfigProto
  - tf.set_random_seed (난수 재현성)
- 설정: CPU/GPU 병렬화, 난수 시드 (lines 69-72)
- 호출 위치: utils.py 초기화 부분, Keras 모델 학습

### 시각화 라이브러리

**matplotlib** (2.0+)
- 용도: 그래프 출력
- 함수: pyplot.figure, pyplot.subplot, pyplot.show, pyplot.xticks
- 호출 위치: plot_numeric_for_regression, plot_categorical_for_regression, LgbmWrapper.plot_importance

**seaborn** (0.8+)
- 용도: 고급 통계 시각화
- 함수: distplot, scatterplot, countplot, boxplot
- 호출 위치: plot_numeric_for_regression, plot_categorical_for_regression

### 유틸리티 라이브러리

**tqdm** (4.20+)
- 용도: 진행 바 표시
- import: tqdm_notebook (Jupyter 환경)
- 호출 위치: (현재 코드에서는 import만 함)

**psutil** (5.4+)
- 용도: 시스템 리소스 모니터링 (메모리, CPU)
- 호출 위치: (현재 코드에서는 import만 함)

## 내부 모듈 의존성

### 노트북 → utils.py 관계

```
Notebook 실행 순서    Import 구조
     ↓
[1] geo-data-eda-and-feature-engineering.ipynb
    ├─ import: from notebook.utils import *
    ├─ 사용 함수:
    │  ├─ load_data() → 데이터 로드 및 피처 엔지니어링
    │  ├─ reduce_mem_usage() → 메모리 최적화
    │  ├─ display_all() → 데이터 출력
    │  ├─ calc_missing_stat() → 결측치 분석
    │  ├─ plot_numeric_for_regression() → 시각화
    │  ├─ plot_categorical_for_regression() → 시각화
    │  └─ rmse() → 평가 지표
    └─ 출력: X_train, X_test (피처), y_train (타겟)

     ↓

[2] Generate Neighbor Info.ipynb
    ├─ import: from notebook.utils import *
    ├─ 사용 함수:
    │  ├─ haversine_array() → 거리 계산
    │  ├─ bearing_array() → 방위각 계산
    │  ├─ groupby_helper() → 이웃 통계 집계
    │  └─ dummy_manhattan_distance() → 맨해튼 거리
    └─ 출력: neighbor_*km_stat.csv (거리 기반 이웃)

     ↓

[3] Generate Neighbor Stat.ipynb
    ├─ import: from notebook.utils import *
    ├─ 사용 함수:
    │  ├─ groupby_helper() → 이웃 특성 집계
    │  ├─ load_data() → 데이터 로드
    │  └─ display_all() → 출력
    └─ 출력: nearest_*_neighbor_stat.csv (K-NN 기반 이웃)

     ↓

[4] Stacking Ensemble.ipynb
    ├─ import: from notebook.utils import *
    ├─ 모델 생성:
    │  ├─ XgbWrapper(params={...}) → XGBoost 모델
    │  ├─ LgbmWrapper(params={...}) → LightGBM 모델
    │  ├─ CatWrapper(params={...}) → CatBoost 모델
    │  ├─ KerasWrapper(model_func, params={...}) → 신경망
    │  ├─ KerasEmbeddingWrapper(model_func, params={...}) → 임베딩 신경망
    │  └─ SklearnWrapper(clf, params={...}) → scikit-learn 모델
    ├─ 사용 함수:
    │  ├─ load_data() → 전처리 데이터 로드
    │  ├─ get_oof() → OOF 예측 생성 (Stage 1)
    │  ├─ stacking() → 메타 특성 생성 (Stage 1 → Stage 2)
    │  ├─ rmse() → 모델 평가
    │  └─ reduce_mem_usage() → 메모리 최적화
    └─ 출력: x_train_stage1/2.csv, x_test_stage1/2.csv, 최종 예측
```

### 함수 호출 그래프

```
get_oof()
├─ KFold() 또는 StratifiedKFold() [scikit-learn]
├─ clf.train() → 각 래퍼의 train 메서드
│  ├─ XgbWrapper.train()
│  │  ├─ xgb.DMatrix() [XGBoost]
│  │  └─ xgb.train() [XGBoost]
│  ├─ LgbmWrapper.train()
│  │  ├─ lgb.Dataset() [LightGBM]
│  │  └─ lgb.train() [LightGBM]
│  ├─ CatWrapper.train()
│  │  ├─ cat.Pool() [CatBoost]
│  │  └─ cat.CatBoost().fit() [CatBoost]
│  ├─ KerasWrapper.train()
│  │  └─ keras.model.fit() [Keras]
│  └─ SklearnWrapper.train()
│     └─ sklearn_model.fit() [scikit-learn]
├─ clf.predict() → 각 래퍼의 predict 메서드
└─ eval_func(y_true, y_pred) [rmse 등]

stacking()
├─ get_oof() [위의 get_oof 호출 그래프]
├─ np.concatenate() [numpy]
└─ pd.DataFrame() [pandas]

load_data()
├─ pd.read_csv() [pandas]
├─ pd.concat(), pd.merge() [pandas]
├─ np.log1p(), np.nan, np.isinf() [numpy]
├─ pd.to_datetime(), dt.* [pandas]
├─ pd.get_dummies() [pandas]
├─ LabelEncoder().fit_transform() [scikit-learn]
├─ PCA() [scikit-learn]
├─ KMeans() [scikit-learn]
├─ haversine_array(), bearing_array() [utils.py 지리 함수]
├─ groupby_helper() [utils.py]
├─ RobustScaler() [scikit-learn]
├─ boxcox1p() [scipy]
└─ reduce_mem_usage() [utils.py]

haversine_array()
└─ np.radians(), np.sin(), np.cos(), np.arcsin() [numpy]

bearing_array()
└─ np.radians(), np.sin(), np.cos(), np.arctan2() [numpy]

groupby_helper()
├─ df.groupby()[target_col].agg() [pandas]
├─ get_prefix() [utils.py]
└─ pd.DataFrame().reset_index() [pandas]
```

## 데이터 파일 의존성

### 입력 파일 의존성

```
input/train.csv
├─ 로드: load_data() [Phase 1]
├─ 구조: id, price, date, bedrooms, bathrooms, sqft_living,
│        sqft_lot, floors, waterfront, view, condition, grade,
│        sqft_above, sqft_basement, yr_built, yr_renovated,
│        zipcode, lat, long, sqft_living15, sqft_lot15
└─ 행 수: ~21,500

input/test.csv
├─ 로드: load_data() [Phase 1]
├─ 구조: id, date, bedrooms, bathrooms, ... (price 제외)
└─ 행 수: ~6,500

input/neighbor_1km_stat.csv
├─ 로드: load_data(nb_1km=True) [Phase 4]
├─ 생성: Generate Neighbor Info.ipynb, Generate Neighbor Stat.ipynb
└─ 특성: nb_1km_*_mean, nb_1km_*_std, 등

input/neighbor_3km_stat.csv
├─ 로드: load_data(nb_3km=True)
└─ 유사 구조

input/neighbor_5km_stat.csv
├─ 로드: load_data(nb_5km=True)
└─ 유사 구조

input/nearest_5_neighbor_stat.csv
├─ 로드: load_data(n_5_nb=True)
├─ 생성: Generate Neighbor Stat.ipynb (K-NN K=5)
└─ 특성: n_5_nb_*_mean, n_5_nb_*_std, 등

input/nearest_10_neighbor_stat.csv
├─ 로드: load_data(n_10_nb=True)
└─ 유사 구조 (K=10)

input/nearest_20_neighbor_stat.csv
├─ 로드: load_data(n_20_nb=True)
└─ 유사 구조 (K=20)

input/x_train_single.csv
├─ 생성: geo-data-eda-and-feature-engineering.ipynb
├─ 로드: Stacking Ensemble.ipynb (load_data 호출에 의해 자동 생성)
├─ 구조: 피처 엔지니어링된 훈련 특성 (1000+ 특성)
└─ 행 수: ~21,400 (이상치 제거 후)

input/x_test_single.csv
├─ 생성: geo-data-eda-and-feature-engineering.ipynb
├─ 구조: 피처 엔지니어링된 테스트 특성
└─ 행 수: ~6,500

input/x_train_stage1.csv
├─ 생성: Stacking Ensemble.ipynb (Stage 1 학습)
├─ 구조: Stage 1 모델의 OOF 예측 (모델 수 = 열 수)
├─ 행 수: ~21,400
└─ 활용: Stage 2 메타 특성으로 사용

input/x_test_stage1.csv
├─ 생성: Stacking Ensemble.ipynb (Stage 1 예측)
├─ 구조: Stage 1 모델의 테스트 예측
└─ 행 수: ~6,500

input/x_train_stage2.csv
├─ 생성: Stacking Ensemble.ipynb (메타 특성 생성)
├─ 구조: x_train_stage1 + 새로운 메타 특성 (선택사항)
└─ 활용: Stage 2 메타 모델 학습

input/x_test_stage2.csv
├─ 생성: Stacking Ensemble.ipynb
└─ 구조: x_test_stage1 + 새로운 메타 특성

input/cv_score_single.csv
├─ 기록: geo-data-eda-and-feature-engineering.ipynb
├─ 내용: 단일 모델의 CV 점수
└─ 형식: [model_name, fold_score_1, ..., fold_score_5, mean_score]

input/cv_score_stage1.csv
├─ 기록: Stacking Ensemble.ipynb (Stage 1)
├─ 내용: 모든 기초 모델의 CV 점수
└─ 형식: [model, cv_score, ...]

input/cv_score_stage2.csv
├─ 기록: Stacking Ensemble.ipynb (Stage 2)
├─ 내용: 메타 모델의 최종 CV 점수
└─ 형식: [model, cv_score]
```

### 의존성 방향

```
Stage 1 (병렬 처리 가능):
train.csv → load_data() → X_train_single, y_train
test.csv                → X_test_single

이웃 특성 추출 (순차):
train.csv + test.csv
    ↓
[Generate Neighbor Info.ipynb]
    ↓
neighbor_*km.csv
    ↓
[Generate Neighbor Stat.ipynb]
    ↓
neighbor_*km_stat.csv, nearest_*_neighbor_stat.csv

Stage 2 (순차):
X_train_single + neighbor 특성
    ↓
[Stacking Ensemble Phase 1]
    ↓
x_train_stage1.csv, x_test_stage1.csv, cv_score_stage1.csv
    ↓
[메타 특성 생성]
    ↓
x_train_stage2.csv, x_test_stage2.csv
    ↓
[Stacking Ensemble Phase 2]
    ↓
최종 예측 + cv_score_stage2.csv
```

## 순환 의존성 (없음)

모든 의존성이 DAG(Directed Acyclic Graph) 구조로 순환 없음 ✓

## 의존성 버전 호환성

**권장 환경**:
- Python 3.5+
- pandas 0.23+
- numpy 1.14+
- scikit-learn 0.19+
- XGBoost 0.70+
- LightGBM 2.1+
- CatBoost 0.8+
- Keras 2.1+
- TensorFlow 1.5+
- scipy 1.0+
- matplotlib 2.0+
- seaborn 0.8+

**주의**: TensorFlow 2.0+ 사용 시 Keras API 호환성 확인 필요 (tf.keras 변경)
