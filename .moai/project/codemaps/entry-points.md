# 진입점 및 실행 가이드

## 노트북 실행 순서

### 전체 파이프라인 실행 흐름

```
Step 1: EDA 및 피처 엔지니어링 (필수)
  ↓
Step 2: 지리적 이웃 정보 추출 (선택적, 권장)
  ↓
Step 3: 이웃 통계 집계 (선택적, 권장)
  ↓
Step 4: 스태킹 앙상블 (필수)
  ↓
최종 제출
```

## Step 1: geo-data-eda-and-feature-engineering.ipynb

### 목표
원본 데이터 로드, 탐색적 데이터 분석, 피처 엔지니어링 수행

### 입력 파일
- `../input/train.csv` - 훈련 데이터 (필수)
- `../input/test.csv` - 테스트 데이터 (필수)

### 주요 코드 구조

```python
# 1단계: 유틸리티 임포트
from notebook.utils import *

# 2단계: 데이터 로드 및 기본 통계
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 3단계: 결측치 및 이상치 분석
calc_missing_stat(train, missing_only=True)
# 결과: feature별 null 개수, 유니크 개수, 타입

# 4단계: EDA 시각화 (선택적)
plot_numeric_for_regression(data, 'sqft_living', 'price')
plot_categorical_for_regression(data, 'grade', 'price')

# 5단계: 피처 엔지니어링 및 전처리
X_train, X_test, y_train = load_data(
    nb_1km=False,      # 이웃 특성 미포함 (Step 3 이후)
    nb_3km=False,
    nb_5km=False,
    n_5_nb=False,
    n_10_nb=False,
    n_20_nb=False,
    original=True,     # 피처 엔지니어링 적용
    do_scale=False,    # 스케일링은 나중에
    fix_skew=False,    # 분포 정정은 나중에
    do_ohe=True        # One-Hot Encoding 적용
)

# 6단계: 메모리 최적화
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

# 7단계: 데이터 저장
X_train.to_csv('../input/x_train_single.csv', index=False)
X_test.to_csv('../input/x_test_single.csv', index=False)
y_train.to_csv('../input/y_train.csv', index=False)

# 8단계: 기본 모델로 베이스라인 평가 (선택적)
from sklearn.linear_model import Ridge
baseline = Ridge(alpha=1.0)
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("Baseline RMSE:", rmse(y_test, baseline_pred))
```

### 출력 파일
- `../input/x_train_single.csv` - 피처 엔지니어링된 훈련 특성
- `../input/x_test_single.csv` - 피처 엔지니어링된 테스트 특성
- `../input/y_train.csv` - 훈련 타겟 (선택적)
- `../input/cv_score_single.csv` - CV 점수 기록

### 피처 엔지니어링 내용

**생성되는 주요 특성 (100+개)**:
- 날짜 특성: yr_mo_sold, yr_sold, qt_sold, week_sold, dow_sold
- 상호작용: bedrooms+bathrooms, sqft_living/bedrooms, sqft_living*grade
- 주소 특성: zipcode-3, zipcode-4, zipcode-5, zipcode-34, zipcode-45
- 좌표: coord_pca1, coord_pca2, coord_cluster, bearing_from_center, qcut_bearing
- 범주형 통계: grade_price_mean, bedrooms_price_mean (One-Hot Encoding 또는 Label Encoding)
- Boolean: has_basement, is_renovated, sqft_living_changed, sqft_lot_changed

### 실행 팁
- 메모리 부족 시: `reduce_mem_usage()` 사용으로 메모리 25-50% 감소
- 빠른 실행: `do_ohe=False`로 Label Encoding 사용 (10배 빠름)
- EDA 스킵: `load_data()` 바로 호출로 5분 안에 완료

---

## Step 2: Generate Neighbor Info.ipynb

### 목표
Haversine 거리를 사용하여 지리적 이웃 정보 추출

### 입력 파일
- `../input/train.csv` - 좌표 데이터
- `../input/test.csv` - 좌표 데이터

### 주요 코드 구조

```python
# 1단계: 유틸리티 임포트
from notebook.utils import haversine_array, dummy_manhattan_distance

# 2단계: 좌표 데이터 로드
data = pd.concat([
    pd.read_csv('../input/train.csv'),
    pd.read_csv('../input/test.csv')
]).reset_index(drop=True)

# 3단계: 이웃 정보 추출 (반경 1km)
neighbors_1km = []
for idx, row in data.iterrows():
    lat1, long1 = row['lat'], row['long']
    distances = haversine_array(lat1, long1, data['lat'].values, data['long'].values)
    neighbor_indices = np.where((distances > 0) & (distances <= 1))[0]
    neighbors_1km.append(neighbor_indices)

# 4단계: 이웃 특성 집계 (나중 단계에서 처리)
# 여기서는 neighbor_indices만 저장

# 5단계: 반복 (3km, 5km도 동일)
# haversine_array로 거리 계산
# 임계값 (1km, 3km, 5km)으로 필터링
# neighbor_indices 저장

# 6단계: K-NN 이웃 추출 (K=5, 10, 20)
from sklearn.neighbors import NearestNeighbors

for K in [5, 10, 20]:
    nbrs = NearestNeighbors(n_neighbors=K+1).fit(data[['lat', 'long']])
    distances, indices = nbrs.kneighbors(data[['lat', 'long']])
    # indices[:, 1:] - 자신 제외한 K개 이웃
    nearest_K_neighbors[K] = indices[:, 1:]

# 7단계: 결과 저장 (정수 인덱스만)
# 다음 단계에서 실제 특성 값 추출
```

### 출력 형식

**neighbor_1km_info.npy** (예시):
- 각 행: [neighbor_indices for 1km radius]
- 형식: numpy array 또는 pickle

**nearest_5_neighbor_info.npy**:
- 각 행: [5 가장 가까운 이웃의 인덱스]
- 형식: (num_samples, 5) 배열

### 주요 함수 호출

```python
# Haversine 거리 벡터화 계산
distances = haversine_array(lat1, long1, data['lat'].values, data['long'].values)

# 맨해튼 거리 근사 (더 빠름)
distances_approx = dummy_manhattan_distance(lat1, long1, data['lat'].values, data['long'].values)

# 방위각 계산 (bearing)
bearings = bearing_array(center_lat, center_long, data['lat'].values, data['long'].values)
```

### 실행 팁
- 대규모 데이터: 청크 단위 처리 (예: 1000행씩)
- 속도 개선: dummy_manhattan_distance 먼저 계산 후 상세 거리 재계산
- 메모리: sparse matrix 사용 고려

---

## Step 3: Generate Neighbor Stat.ipynb

### 목표
Step 2에서 추출한 이웃 정보를 기반으로 이웃 특성 통계 생성

### 입력 파일
- `../input/train.csv` - 원본 특성 데이터
- `../input/test.csv` - 원본 특성 데이터
- 이웃 인덱스 (Step 2 결과)

### 주요 코드 구조

```python
# 1단계: 유틸리티 임포트
from notebook.utils import groupby_helper, load_data

# 2단계: 원본 데이터 로드
data = pd.concat([
    pd.read_csv('../input/train.csv'),
    pd.read_csv('../input/test.csv')
]).reset_index(drop=True)

# 3단계: 반경 기반 통계 (1km, 3km, 5km)
def compute_neighbor_stats_by_distance(data, radius_km, suffix):
    """
    각 샘플별로 반경 내 이웃의 특성 통계 계산
    """
    neighbor_stats = pd.DataFrame({'id': data['id']})

    for idx, row in data.iterrows():
        lat, long = row['lat'], row['long']
        # haversine_array로 거리 계산
        distances = haversine_array(lat, long, data['lat'].values, data['long'].values)

        # 반경 내 이웃 필터링 (자신 제외)
        neighbor_mask = (distances > 0) & (distances <= radius_km)
        neighbor_indices = np.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            neighbors = data.iloc[neighbor_indices]
            # 이웃 특성별 통계 계산
            for feat in ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'view', 'condition']:
                neighbor_stats.loc[idx, f'nb_{suffix}_{feat}_mean'] = neighbors[feat].mean()
                neighbor_stats.loc[idx, f'nb_{suffix}_{feat}_std'] = neighbors[feat].std()
                neighbor_stats.loc[idx, f'nb_{suffix}_{feat}_min'] = neighbors[feat].min()
                neighbor_stats.loc[idx, f'nb_{suffix}_{feat}_max'] = neighbors[feat].max()
        else:
            # 이웃 없으면 0 (결측치)
            neighbor_stats.fillna(0, inplace=True)

    return neighbor_stats

# 4단계: 각 반경별 통계 계산
for radius, suffix in [(1, '1km'), (3, '3km'), (5, '5km')]:
    stats_df = compute_neighbor_stats_by_distance(data, radius, suffix)
    stats_df.to_csv(f'../input/neighbor_{suffix}_stat.csv', index=False)

# 5단계: K-NN 기반 통계
def compute_knn_stats(data, K, neighbors_indices):
    """
    K-NN 이웃의 특성 통계 계산
    neighbors_indices: (num_samples, K) 배열
    """
    knn_stats = pd.DataFrame({'id': data['id']})

    for idx, neighbor_ids in enumerate(neighbors_indices):
        neighbors = data.iloc[neighbor_ids]
        for feat in ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'view', 'condition']:
            knn_stats.loc[idx, f'n_{K}_nb_{feat}_mean'] = neighbors[feat].mean()
            knn_stats.loc[idx, f'n_{K}_nb_{feat}_std'] = neighbors[feat].std()

    return knn_stats

# 6단계: K=5, 10, 20에 대해 계산
for K in [5, 10, 20]:
    stats_df = compute_knn_stats(data, K, nearest_K_neighbors[K])
    stats_df.to_csv(f'../input/nearest_{K}_neighbor_stat.csv', index=False)

# 7단계: 생성된 이웃 특성 검증
neighbor_1km = pd.read_csv('../input/neighbor_1km_stat.csv')
print(neighbor_1km.head())
print(neighbor_1km.shape)  # (num_samples, num_stats)
```

### 출력 파일
- `../input/neighbor_1km_stat.csv` - 1km 반경 이웃 통계
- `../input/neighbor_3km_stat.csv` - 3km 반경 이웃 통계
- `../input/neighbor_5km_stat.csv` - 5km 반경 이웃 통계
- `../input/nearest_5_neighbor_stat.csv` - 5-NN 통계
- `../input/nearest_10_neighbor_stat.csv` - 10-NN 통계
- `../input/nearest_20_neighbor_stat.csv` - 20-NN 통계

### 생성되는 특성 예시

각 파일의 특성:
- `nb_1km_sqft_living_mean` - 1km 반경 이웃의 평균 면적
- `nb_1km_bedrooms_std` - 1km 반경 이웃의 침실 수 표준편차
- `n_5_nb_grade_mean` - 5-NN 이웃의 평균 등급
- ... 등 (총 100+ 개 특성)

### 실행 팁
- 속도: 병렬 처리 (joblib.Parallel) 사용으로 10배 가속
- 메모리: 청크 단위 처리
- 검증: 통계값이 -1~100 범위 내인지 확인

---

## Step 4: Stacking Ensemble.ipynb

### 목표
다양한 기초 모델 학습 후 메타 특성 생성 및 최종 예측

### 입력 파일
- `../input/x_train_single.csv` - 피처 엔지니어링된 훈련 특성
- `../input/x_test_single.csv` - 피처 엔지니어링된 테스트 특성
- `../input/neighbor_*km_stat.csv` - 이웃 통계 (선택적)
- `../input/nearest_*_neighbor_stat.csv` - K-NN 통계 (선택적)

### 주요 코드 구조

```python
# 1단계: 유틸리티 및 모델 임포트
from notebook.utils import *
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 2단계: 전처리 데이터 로드 (이웃 특성 포함)
X_train, X_test, y_train = load_data(
    nb_1km=True, nb_3km=True, nb_5km=True,
    n_5_nb=True, n_10_nb=True, n_20_nb=True,
    original=True, do_scale=True, fix_skew=True, do_ohe=True
)

# 3단계: 기초 모델 정의 (Stage 1)
base_models = [
    ('xgb', XgbWrapper({'learning_rate': 0.05, 'max_depth': 5}, num_rounds=500)),
    ('lgb', LgbmWrapper({'learning_rate': 0.05, 'max_depth': 5}, num_rounds=500)),
    ('cat', CatWrapper({'learning_rate': 0.05, 'depth': 5}, num_rounds=500)),
    ('ridge', SklearnWrapper(Ridge, {'alpha': 100})),
    ('lasso', SklearnWrapper(Lasso, {'alpha': 1})),
    ('elastic', SklearnWrapper(ElasticNet, {'alpha': 1, 'l1_ratio': 0.5})),
    ('rf', SklearnWrapper(RandomForestRegressor, {'n_estimators': 100, 'random_state': 0})),
]

# 4단계: Stage 1 - 기초 모델 학습 (OOF 생성)
oof_train_list = []
oof_test_list = []
cv_scores = []

for name, model in base_models:
    print(f"\n=== Training {name} ===")

    oof_train, oof_test, cv_score = get_oof(
        model, X_train, y_train, X_test, rmse,
        NFOLDS=5, kfold_random_state=0
    )

    oof_train_list.append(oof_train)
    oof_test_list.append(oof_test)
    cv_scores.append({
        'model': name,
        'cv_score': cv_score,
        'oof_shape': oof_train.shape
    })

# 5단계: Stage 1 결과 저장
X_train_stage1 = pd.DataFrame(
    np.concatenate(oof_train_list, axis=1),
    columns=[f'model_{i}' for i in range(len(base_models))]
)
X_test_stage1 = pd.DataFrame(
    np.concatenate(oof_test_list, axis=1),
    columns=[f'model_{i}' for i in range(len(base_models))]
)

X_train_stage1.to_csv('../input/x_train_stage1.csv', index=False)
X_test_stage1.to_csv('../input/x_test_stage1.csv', index=False)

cv_score_df = pd.DataFrame(cv_scores)
cv_score_df.to_csv('../input/cv_score_stage1.csv', index=False)

# 6단계: Stage 2 - 메타 특성 생성
X_train_stage2 = X_train_stage1.copy()  # 또는 추가 특성 병합
X_test_stage2 = X_test_stage1.copy()

X_train_stage2.to_csv('../input/x_train_stage2.csv', index=False)
X_test_stage2.to_csv('../input/x_test_stage2.csv', index=False)

# 7단계: Stage 2 - 메타 모델 학습
meta_model = XgbWrapper(
    {'learning_rate': 0.05, 'max_depth': 5},
    num_rounds=500, early_stopping=100
)

oof_final, pred_final, final_score = get_oof(
    meta_model, X_train_stage2, y_train, X_test_stage2, rmse,
    NFOLDS=5, kfold_random_state=0
)

print(f"\nFinal CV Score: {final_score}")

# 8단계: 최종 예측 (로그 스케일 → 원본 스케일)
pred_final_exp = np.expm1(pred_final)
submission = pd.DataFrame({
    'id': pd.read_csv('../input/test.csv')['id'],
    'price': pred_final_exp.flatten()
})

submission.to_csv('submission.csv', index=False)

# 9단계: 결과 저장
x_train_second = oof_final
x_test_second = pred_final
x_train_second.to_csv('../input/x_train_second.csv', index=False)
x_test_second.to_csv('../input/x_test_second.csv', index=False)

final_scores = pd.DataFrame({
    'stage': ['stage2'],
    'cv_score': [final_score]
})
final_scores.to_csv('../input/cv_score_stage2.csv', index=False)
```

### 출력 파일
- `submission.csv` - 최종 제출 파일 (id, price)
- `../input/x_train_stage1.csv` - Stage 1 OOF 예측
- `../input/x_test_stage1.csv` - Stage 1 테스트 예측
- `../input/x_train_stage2.csv` - Stage 2 입력 특성
- `../input/x_test_stage2.csv` - Stage 2 입력 특성
- `../input/x_train_second.csv` - 최종 OOF 예측
- `../input/x_test_second.csv` - 최종 테스트 예측
- `../input/cv_score_stage1.csv` - 기초 모델 CV 점수
- `../input/cv_score_stage2.csv` - 메타 모델 CV 점수

### Stage 1 vs Stage 2

**Stage 1 (기초 모델)**:
- 입력: 1000+ 개 원본 특성
- 모델: 6-10개 다양한 알고리즘 (XGBoost, LightGBM, CatBoost, Ridge, Lasso 등)
- 출력: 각 모델의 OOF 예측 (6-10 개 열)
- 목표: 다양한 관점의 예측 생성

**Stage 2 (메타 모델)**:
- 입력: 6-10 개 메타 특성 (Stage 1 OOF)
- 모델: 1개 메타 모델 (XGBoost 권장)
- 출력: 최종 예측
- 목표: 기초 모델 예측을 최적으로 결합

### 실행 팁

**속도 최적화**:
- `num_rounds` 줄이기 (500 → 300)
- 모델 개수 줄이기 (10 → 6)
- 병렬 처리 (XGBoost n_jobs=-1)

**정확도 최적화**:
- `num_rounds` 늘리기 (500 → 1000)
- K-Fold 수 증가 (5 → 10)
- 모델 다양성 증가 (신경망 추가)

**메모리 최적화**:
- `reduce_mem_usage()` 사용
- 중간 결과 del로 메모리 해제
- gc.collect() 호출

---

## 전체 실행 명령어

```bash
# Step 1: 피처 엔지니어링 (30~60분)
jupyter notebook notebook/geo-data-eda-and-feature-engineering.ipynb

# Step 2: 이웃 정보 추출 (10~20분)
jupyter notebook notebook/Generate\ Neighbor\ Info.ipynb

# Step 3: 이웃 통계 계산 (20~40분)
jupyter notebook notebook/Generate\ Neighbor\ Stat.ipynb

# Step 4: 스태킹 앙상블 (2~4시간, GPU 권장)
jupyter notebook notebook/Stacking\ Ensemble.ipynb
```

## 병렬 실행 가능 여부

**병렬 가능**:
- Step 1과 Step 2/3 일부: 데이터 독립적이면 병렬 처리 가능

**순차 필수**:
- Step 2 → Step 3 (Step 2 이웃 정보 필요)
- Step 3 → Step 4 (Step 3 이웃 특성 필요)

**권장 순서**: Step 1 → (Step 2 + Step 3) → Step 4
