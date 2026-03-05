# 모듈 설명

## utils.py 구조 (842줄)

### 전체 구성

```
imports
  ├─ 표준 라이브러리: sys, os, re, math, json, pickle, time, gc, etc
  ├─ 데이터 처리: pandas, numpy, scipy
  ├─ 머신러닝: sklearn, xgboost, lightgbm, catboost
  ├─ 딥러닝: keras, tensorflow
  └─ 시각화: matplotlib, seaborn

난수 시드 설정 (RANDOM_SEED = 0)
  └─ 재현성 보장

헬퍼 함수 (유틸리티)
모델 래퍼 클래스 (5개)
앙상블 함수
데이터 로딩 및 전처리 함수
지리 거리 계산 함수
```

## 주요 클래스

### 1. SklearnWrapper (220-238줄)

**목적**: scikit-learn 모델을 통일된 인터페이스로 래핑

**메서드**:
- `__init__(clf, params=None, **kwargs)`
  - clf: scikit-learn 클래스 (Ridge, Lasso, RandomForest, etc)
  - params: 모델 파라미터 dict
  - use_avg_oof: 테스트 예측 평균 여부

- `train(x_train, y_train, x_cross=None, y_cross=None)` [@time_decorator]
  - 모델 학습
  - 유니크 타겟 값 30개 기준으로 분류/회귀 자동 판정
  - 분류: predict_proba() 사용, 회귀: predict() 사용

- `predict(x)` -> numpy array
  - 분류 문제: 확률 (클래스 1의 확률)
  - 회귀 문제: 예측값

**사용 예**:
```
wrapper = SklearnWrapper(Ridge, {'alpha': 1.0})
wrapper.train(X_train, y_train, X_valid, y_valid)
pred = wrapper.predict(X_test)
```

### 2. XgbWrapper (240-282줄)

**목적**: XGBoost 모델을 스태킹에 최적화된 래퍼로 제공

**파라미터**:
- `params`: XGBoost 학습 파라미터 dict
- `num_rounds`: 부스팅 라운드 수 (기본값: 1000)
- `early_stopping`: 조기 종료 라운드 (기본값: 100)
- `eval_function`: 평가 함수 (RMSE 등)
- `verbose_eval`: 평가 출력 간격

**메서드**:
- `train(x_train, y_train, x_cross=None, y_cross=None)` [@time_decorator]
  - x_cross 없음: 모든 데이터로 num_rounds 학습
  - x_cross 있음: 검증 데이터 사용 조기 종료 학습
  - best_round 자동 기록

- `predict(x)` -> numpy array
  - best_round까지의 누적 예측 반환

- `get_params()` -> dict

**특징**: 조기 종료로 최적 라운드 자동 탐색

### 3. LgbmWrapper (284-336줄)

**목적**: LightGBM 모델 래핑

**파라미터**: XgbWrapper와 유사
- `params`, `num_rounds`, `early_stopping`, `eval_function`, `verbose_eval`

**메서드**:
- `train(x_train, y_train, x_cross=None, y_cross=None)` [@time_decorator]
  - x_cross 있으면 검증 세트 사용 학습
  - best_iteration 자동 기록
  - 메모리 해제 (gc.collect())

- `predict(x)` -> numpy array
  - best_iteration까지 예측

- `plot_importance(importance_type='gain', max_num_features=20)`
  - 피처 중요도 시각화

- `get_params()` -> dict

**특징**: 빠른 학습 속도, 피처 중요도 제공

### 4. CatWrapper (338-389줄)

**목적**: CatBoost 모델 래핑 (범주형 특성 최적화)

**메서드**:
- `train(x_train, y_train, x_cross=None, y_cross=None, cat_features=None)` [@time_decorator]
  - cat_features: 범주형 특성 인덱스 리스트
  - cat.Pool 사용 데이터 래핑
  - 검증 세트 선택적 사용

- `predict(x)` -> numpy array
  - best_iteration_ 기준 예측

- `get_params()` -> dict

**특징**: 범주형 변수 자동 처리, Pool 기반 데이터 관리

### 5. KerasWrapper (391-424줄)

**목적**: Keras 신경망 모델 래핑

**파라미터**:
- `model_func`: 신경망 생성 함수 (입력 차원 받음)
- `epochs`: 에포크 수 (기본값: 20)
- `batch_size`: 배치 크기 (기본값: 16)
- `callbacks`: Keras 콜백 리스트 (EarlyStopping, ModelCheckpoint 등)

**메서드**:
- `train(x_train, y_train, x_cross=None, y_cross=None)` [@time_decorator]
  - model_func(x_train.shape[1])로 모델 생성
  - x_cross 없음: best_epochs로 학습
  - x_cross 있음: 검증 데이터 사용, best_epochs 업데이트

- `predict(x)` -> numpy array
  - 1D 배열로 평탄화 (.ravel())

**특징**: 신경망 모델 통합, 조기 종료 지원

### 6. KerasEmbeddingWrapper (426-475줄)

**목적**: 임베딩 레이어 기반 신경망 래핑 (범주형 변수 처리)

**파라미터**:
- `model_func`: 신경망 생성 함수 (x_train, embedding_cols 받음)
- `embedding_cols`: 임베딩할 범주형 특성 리스트

**메서드**:
- `train(x_train, y_train, x_cross=None, y_cross=None)` [@time_decorator]
  - 수치형 특성과 임베딩 특성을 분리
  - 리스트 형식으로 모델에 전달
  - 검증 데이터도 동일 형식 처리

- `predict(x)` -> numpy array
  - 입력을 리스트로 변환하여 예측

**특징**: 범주형 변수 임베딩 자동화

## 주요 함수

### 데이터 처리 헬퍼

**display_all(df)** (74-76줄)
- pandas 디스플레이 옵션 일시 설정
- 큰 데이터프레임 전체 표시

**reduce_mem_usage(df)** (78-114줄)
- 데이터타입 최적화로 메모리 감소
- int8/16/32/64, float16/32/64 자동 선택
- 범주형 변수는 category로 변환
- 메모리 감소율 출력

**calc_missing_stat(df, missing_only=True)** (117-126줄)
- 결측치 통계 분석
- 반환: feature, num_of_unique, pct_of_missing, type

### 시각화 함수

**plot_numeric_for_regression(df, field, target_field='price')** (128-147줄)
- 수치형 변수 분포 및 타겟과의 관계 시각화
- 좌: 훈련/테스트 분포, 우: 산점도

**plot_categorical_for_regression(df, field, target_field='price', ...)** (149-175줄)
- 범주형 변수 카운트 및 타겟과의 박스플롯 시각화
- 카테고리 > 30개면 2x1 레이아웃

### 집계 헬퍼

**get_prefix(group_col, target_col, prefix=None)** (177-188줄)
- 그룹화된 특성 이름 생성
- 예: "grade_price_mean" → "grade_price" + "mean"

**groupby_helper(df, group_col, target_col, agg_method, prefix_param=None)** (190-198줄)
- 그룹별 집계 수행
- 입력: group_col (그룹 기준), target_col (집계 대상), agg_method (mean, std, etc)
- 반환: 집계 결과 데이터프레임

### 평가 지표

**rmse(y_true, y_pred)** (200-201줄)
- 제곱근 평균 제곱 오차 계산

**rmse_exp(y_true, y_pred)** (203-204줄)
- 지수 변환 후 RMSE 계산 (로그 변환된 데이터용)

### 데코레이터

**time_decorator(func)** (206-218줄)
- 함수 실행 시간 측정
- 시작/종료 시간 및 총 소요 시간 출력
- KST 시간 기준 출력

## 앙상블 함수

### get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs) (478-542줄)

**목적**: K-Fold CV를 통해 Out-Of-Fold(OOF) 예측 생성

**파라미터**:
- `clf`: 모델 래퍼 객체 (XgbWrapper, LgbmWrapper 등)
- `x_train, y_train`: 훈련 데이터
- `x_test`: 테스트 데이터
- `eval_func`: 평가 함수 (rmse 등)
- `NFOLDS`: 폴드 수 (기본값: 5)
- `kfold_shuffle`: 섞기 여부 (기본값: True)
- `kfold_random_state`: 난수 시드 (기본값: 0)
- `stratifed_kfold_y_value`: 계층화 기준 (None이면 KFold, 아니면 StratifiedKFold)

**반환**:
- `oof_train`: (ntrain, 1) 배열 - 훈련 데이터 OOF 예측
- `oof_test`: (ntest, 1) 배열 - 테스트 데이터 예측
- `score`: float - 평균 CV 점수

**동작**:
1. K-Fold 분할 (또는 StratifiedKFold)
2. 각 폴드마다:
   - 훈련 세트로 모델 학습
   - 검증 세트에서 예측
   - CV 점수 기록
3. 모델이 use_avg_oof=True면 테스트 예측 평균화
4. use_avg_oof=False면 전체 데이터로 재학습 후 테스트 예측

**특징**:
- OOF를 메타 특성으로 활용 가능
- CV 점수를 통해 모델 성능 평가

### stacking(data_list, y_train, model_list, eval_func=None, nfolds=5, ...) (544-565줄)

**목적**: 여러 데이터셋에 여러 모델로 스태킹 수행

**파라미터**:
- `data_list`: [(X_train1, X_test1), (X_train2, X_test2), ...] 리스트
- `y_train`: 훈련 타겟
- `model_list`: [XgbWrapper(), LgbmWrapper(), ...] 모델 리스트
- `eval_func`: 평가 함수
- `nfolds`: 폴드 수

**반환**:
- `X_train_next`: (ntrain, num_models * num_datasets) 메타 특성
- `X_test_next`: (ntest, num_models * num_datasets) 메타 특성
- `oof_cv_score_list`: 모든 모델의 CV 점수 리스트

**동작**:
1. 각 데이터셋과 모델 조합마다 get_oof() 실행
2. 모든 OOF 예측을 열 방향으로 연결
3. 메타 특성 생성 (다음 단계 입력)

**활용**: Stage 1 → Stage 2 메타 특성 생성

## 데이터 로딩 함수

### load_data(nb_1km=True, nb_3km=True, nb_5km=True, n_5_nb=True, n_10_nb=True, n_20_nb=True, original=True, do_scale=False, fix_skew=False, do_ohe=True) (568-819줄)

**목적**: 원본 데이터 + 이웃 특성을 로드하여 전처리된 X_train, X_test, y_train 반환

**파라미터**:
- `nb_*km`: 반경 기반 이웃 특성 포함 여부 (1km, 3km, 5km)
- `n_*_nb`: K-NN 기반 이웃 특성 포함 여부 (5, 10, 20)
- `original`: 원본 피처 엔지니어링 여부
- `do_scale`: RobustScaler 적용 여부
- `fix_skew`: Box-Cox 변환으로 분포 정정 여부
- `do_ohe`: One-Hot Encoding 여부 (아니면 LabelEncoder)

**주요 처리**:

1. **데이터 로드**: train.csv, test.csv 로드
2. **이상치 제거**: sqft_living > 12000 & price < 3000000
3. **로그 변환**: price → log1p(price)
4. **피처 엔지니어링** (original=True일 때):
   - 날짜 특성 추출 (년, 월, 분기, 주, 요일)
   - 수치형 상호작용 (비율, 곱셈, 차이)
   - 주소 코드 분해 (zipcode substring)
   - 좌표 기반 변환 (PCA, K-means, bearing)
   - 범주형 통계 (mean price by group)
5. **이웃 특성 병합** (nb_*, n_*_nb=True일 때):
   - neighbor_*km_stat.csv 로드 및 병합
   - 이웃 특성 차이 생성 (자신 - 이웃 평균)
6. **분포 정정** (fix_skew=True일 때):
   - 왜도 > 0.75인 특성에 Box-Cox 변환
7. **스케일링** (do_scale=True일 때):
   - RobustScaler로 수치형 특성 정규화
8. **원-핫 인코딩**: 범주형 변수 처리

**반환**: (X_train, X_test, y_train)

**특징**:
- 유연한 옵션으로 다양한 특성 조합 실험 가능
- 메모리 효율성을 위해 dtype 최적화

## 지리 계산 함수

### haversine_array(lat1, lng1, lat2, lng2) (821-828줄)

**목적**: 라디안 단위 좌표 간 Haversine 거리 계산

**공식**: 대원 거리 계산 (지구 반지름 = 6371km)

**입력**: 도(degree) 단위 좌표 (자동 라디안 변환)

**반환**: 거리 (km)

### dummy_manhattan_distance(lat1, lng1, lat2, lng2) (830-833줄)

**목적**: 더미 맨해튼 거리 (위도 거리 + 경도 거리)

**용도**: 빠른 근사 거리 계산

### bearing_array(lat1, lng1, lat2, lng2) (835-841줄)

**목적**: 한 점에서 다른 점으로의 방위각 계산

**반환**: 도(degree) 단위 각도 (-180 ~ 180)

**활용**: 중심 좌표 기준 방향 특성

## 공개 인터페이스

### 모델 학습 인터페이스

```python
# 모델 생성
model = XgbWrapper(params={'learning_rate': 0.1, ...}, num_rounds=1000)

# 학습
model.train(X_train, y_train, X_valid, y_valid)

# 예측
pred = model.predict(X_test)
```

### OOF 생성 인터페이스

```python
# 단일 모델 OOF
oof_train, oof_test, cv_score = get_oof(
    model, X_train, y_train, X_test, rmse,
    NFOLDS=5, kfold_random_state=0
)

# 다중 모델 스태킹
X_train_meta, X_test_meta, scores = stacking(
    data_list=[(X_train, X_test)],
    y_train=y_train,
    model_list=[xgb_model, lgb_model, cat_model],
    eval_func=rmse,
    nfolds=5
)
```

### 데이터 로딩 인터페이스

```python
# 전체 특성 포함
X_train, X_test, y_train = load_data(
    nb_1km=True, nb_3km=True, nb_5km=True,
    n_5_nb=True, n_10_nb=True, n_20_nb=True,
    original=True, do_scale=True, fix_skew=True
)

# 기본 특성만
X_train, X_test, y_train = load_data(
    original=True, nb_1km=False, n_5_nb=False, ...
)
```
