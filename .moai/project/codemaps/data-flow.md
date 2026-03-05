# 데이터 흐름 분석

## 전체 데이터 파이프라인

### Phase 1: 데이터 로드 및 기본 전처리

```
원본 파일: train.csv, test.csv
  ↓
pandas.read_csv()
  ├─ train: (21,613 행, 21 열) [id, price, date, bedrooms, ...]
  └─ test: (6,498 행, 20 열) [id, date, bedrooms, ..., price 없음]
  ↓
메모리 최적화: reduce_mem_usage()
  ├─ dtype 최적화 (int32 → int8, float64 → float32)
  └─ 메모리: 원본 ~15MB → ~7MB (50% 감소)
  ↓
이상치 제거
  ├─ 조건: (sqft_living > 12000) & (price < 3000000)
  ├─ 제거: 74행 (고급 대형 저가 주택)
  └─ train: 21,613 → 21,539행
  ↓
결측치 분석: calc_missing_stat()
  ├─ null 개수, 비율, 데이터 타입 집계
  └─ 로그 변환 대비 결측 패턴 파악
  ↓
데이터프레임 병합: pd.concat([train_copy, test_copy])
  ├─ 목적: train/test 함께 전처리하여 피처 분포 일치
  ├─ 구조: train 21,539행 + test 6,498행 = 28,037행
  └─ data 컬럼 추가: 'train' 또는 'test'
```

### Phase 2: 피처 엔지니어링 (원본 특성 생성)

```
입력: 기본 전처리 데이터 (21개 원본 특성)
  ↓
[2-1] 타겟 변수 변환
  ├─ np.log1p(price) → log_price
  ├─ 목적: 거래 가격이 우편향 분포이므로 로그 정규화
  └─ RMSE 계산도 지수 변환하여 원본 스케일로 평가
  ↓
[2-2] 날짜 특성 추출
  ├─ date → pd.to_datetime()
  ├─ 생성 특성:
  │  ├─ yr_mo_sold (string) - "YYYY-MM" 형식
  │  ├─ yr_sold (categorical) - 연도
  │  ├─ qt_sold (categorical) - 분기 (1~4)
  │  ├─ week_sold (categorical) - 주차 (1~53)
  │  └─ dow_sold (categorical) - 요일 (0=월요일, 6=일요일)
  ├─ 활용: 계절성, 시장 트렌드 포착
  └─ 범주형 변수화 → One-Hot Encoding 또는 Label Encoding
  ↓
[2-3] 연도 관련 특성
  ├─ yr_sold - yr_built → 건물 나이
  ├─ yr_sold - yr_renovated → 리모델링 이후 연수
  └─ yr_renovated - yr_built → 건물 원래 나이
  ↓
[2-4] 수치형 상호작용 특성
  ├─ bedrooms + bathrooms → 총 방 개수
  ├─ bathrooms / bedrooms → 방당 욕실 비율
  ├─ sqft_living / bedrooms → 침실당 면적
  ├─ sqft_living / bathrooms → 욕실당 면적
  ├─ sqft_living / floors → 층당 면적
  ├─ sqft_lot / sqft_living → 부지 활용도
  ├─ sqft_basement / sqft_above → 지하 비율
  ├─ sqft_lot15 / sqft_living15 → 지역 평균 상대 부지
  └─ 처리: np.isinf() → 0 (0으로 나누기 방지)
  ↓
[2-5] Boolean 특성
  ├─ has_basement = sqft_basement > 0
  ├─ is_renovated = yr_renovated > 0
  ├─ sqft_living_changed = sqft_living != sqft_living15
  ├─ sqft_lot_changed = sqft_lot != sqft_lot15
  └─ 용도: 범주형 특성으로 모델 학습
  ↓
[2-6] 복합 특성
  ├─ sqft_living * grade → 면적 × 품질 복합 지수
  ├─ overall = grade + view + condition + waterfront + has_basement + is_renovated
  │  └─ 종합 품질 점수 (최대 6 + 4 + 5 + 1 + 1 + 1 = 18)
  └─ sqft_living * overall → 면적 × 종합 품질
  ↓
[2-7] 주소 기반 특성
  ├─ zipcode (categorical) - 5자리 우편번호
  ├─ zipcode-3 (categorical) - 3번째 자리
  ├─ zipcode-4 (categorical) - 4번째 자리
  ├─ zipcode-5 (categorical) - 5번째 자리
  ├─ zipcode-34 (categorical) - 3~4번째
  ├─ zipcode-45 (categorical) - 4~5번째
  └─ zipcode-35 (categorical) - 3, 5번째 조합
  ├─ 용도: 지역 기반 가격 모델링
  └─ 관찰: 지역이 주택 가격 최대 영향 요소
  ↓
[2-8] 좌표 변수 차원 축소 (PCA)
  ├─ 입력: lat, long (2D 좌표)
  ├─ PCA(n_components=2)
  ├─ 출력: coord_pca1, coord_pca2 (주성분)
  └─ 목적: 지리 정보 압축 및 노이즈 제거
  ↓
[2-9] 좌표 기반 클러스터링 (K-means)
  ├─ 입력: lat, long
  ├─ KMeans(n_clusters=72, random_state=0)
  │  └─ 72개 지리 영역으로 분할
  ├─ 출력: coord_cluster (0~71, string화 후 'cluster_00' ~ 'cluster_71')
  ├─ 용도: 지역 기반 특성 (One-Hot Encoding)
  └─ 관찰: 같은 클러스터 내 주택은 유사 가격대
  ↓
[2-10] 중심점 기준 방위각 (Bearing)
  ├─ 중심점: (median(lat), median(long)) - 전체 데이터 지리적 중심
  ├─ bearing_array() - 중심에서 각 주택으로의 방위각
  ├─ 출력: bearing_from_center (-180 ~ 180도)
  ├─ pd.qcut(10) - 10개 동일 개수 구간으로 양자화
  │  └─ qcut_bearing (0~9, categorical)
  └─ 용도: 방향별 시장 특성 포착
  ↓
[2-11] 범주형 변수 통계 (타겟 기반)
  ├─ 그룹 기준: grade, bedrooms, bathrooms, view, condition, waterfront
  ├─ groupby_helper()로 각 그룹의 mean(price) 계산
  ├─ 예: grade별 평균 가격 추가
  │  └─ grade_price_mean (카테고리별 평균값)
  ├─ 용도: 범주 자체의 예측력 직접 인코딩
  └─ 위험: 데이터 누수 방지 (훈련 데이터만 학습)
  ↓
[2-12] 분포 정정 (선택적: fix_skew=True)
  ├─ 왜도 계산: scipy.stats.skew()
  ├─ 필터링: |skew| > 0.75인 특성만 변환
  ├─ Box-Cox 변환: scipy.special.boxcox1p(feature, lambda=0.15)
  │  └─ y' = ((y+1)^lambda - 1) / lambda
  └─ 목적: 극심한 분포를 정규분포에 가깝게 정정
  ↓
[2-13] 범주형 변수 인코딩
  ├─ One-Hot Encoding (do_ohe=True):
  │  ├─ pd.get_dummies(df[col], prefix='ohe_'+col)
  │  ├─ 예: yr_sold={2010, 2011, 2012} → 3개 이진 열
  │  └─ 특성 개수: ~200개 증가 (범주 수에 따라)
  └─ Label Encoding (do_ohe=False):
     ├─ sklearn.preprocessing.LabelEncoder()
     ├─ 카테고리 → 정수 (0, 1, 2, ...)
     └─ 특성 개수: 유지 (메모리 절감, 속도 향상)
  ↓
출력: X_train_single, X_test_single (1000+ 특성)
```

### Phase 3: 지리적 이웃 특성

```
입력: lat, long 좌표 (28,037행)
  ↓
[3-1] 거리 계산 (Haversine)
  ├─ haversine_array(lat1, long1, lat2_array, long2_array)
  ├─ 공식: d = 2*R*arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlong/2)))
  │  └─ R = 6371km (지구 반지름)
  ├─ 벡터화 연산: 1행 vs 전체 28,037행 거리 계산
  ├─ 복잡도: O(n) per sample (전체 O(n²))
  └─ 결과: 거리 배열 (km 단위)
  ↓
[3-2] 반경 기반 이웃 추출 (1km, 3km, 5km)
  ├─ 각 주택마다:
  │  ├─ 모든 다른 주택과의 거리 계산
  │  ├─ (거리 > 0) & (거리 <= radius) 필터링
  │  └─ 이웃 인덱스 리스트
  ├─ 통계: 1km 반경에 평균 20~50개 이웃
  └─ 이웃 없는 경우: 0으로 채우기 (NaN 방지)
  ↓
[3-3] K-NN 기반 이웃 추출 (K=5, 10, 20)
  ├─ sklearn.neighbors.NearestNeighbors
  ├─ 각 주택마다 가장 가까운 K개 이웃 추출
  │  ├─ 자신 제외 (거리=0)
  │  └─ K개 선택
  └─ 복잡도: O(n*K*log n) (KDTree 사용)
  ↓
출력: neighbor_*km_info, nearest_*_neighbor_info (이웃 인덱스)
```

### Phase 4: 이웃 특성 통계

```
입력: 이웃 인덱스 + 원본 특성 (X_train_single, X_test_single)
  ↓
[4-1] 각 주택별 반경 내 이웃 특성 집계
  ├─ 주택 i에 대해:
  │  ├─ 1km 반경 이웃 j1, j2, ... 추출
  │  ├─ 이웃들의 sqft_living 값 수집
  │  ├─ 통계 계산:
  │  │  ├─ mean(sqft_living[j1, j2, ...])
  │  │  ├─ std(sqft_living[j1, j2, ...])
  │  │  ├─ min, max, median 등
  │  └─ 결과: nb_1km_sqft_living_mean, nb_1km_sqft_living_std, ...
  ├─ 반복: bedrooms, bathrooms, grade, view, condition (5개 특성)
  ├─ 반복: 1km, 3km, 5km (3개 반경)
  └─ 총 생성 특성: 5 * 4통계 * 3반경 = 60개
  ↓
[4-2] K-NN 이웃 특성 통계
  ├─ 주택 i에 대해:
  │  ├─ K-NN 이웃 선택 (자신 제외)
  │  ├─ 이웃 특성 값 수집
  │  └─ 통계 계산
  ├─ K = 5, 10, 20 (3개 설정)
  └─ 생성 특성: 5 * 4통계 * 3K = 60개
  ↓
[4-3] 이웃과의 차이 특성 (선택적)
  ├─ 목적: 주택 자신의 특성이 지역 평균과 얼마나 다른지
  ├─ 차이 계산: 자신의 특성 - 이웃 평균
  │  ├─ sqft_living - nb_1km_sqft_living_mean
  │  ├─ bedrooms - nb_1km_bedrooms_mean
  │  └─ ...
  ├─ 생성 특성: 5 * 3반경 + 5 * 3K = 30개
  └─ 해석: 양수면 지역 평균보다 우수, 음수면 열등
  ↓
출력: neighbor_1km_stat.csv, ..., nearest_20_neighbor_stat.csv
  └─ 각 파일: (28,037행, 60~120열)
```

### Phase 5: 데이터 통합 및 스케일링

```
입력: x_train_single + 이웃 특성 여러 파일
  ↓
[5-1] 모든 이웃 특성 파일 병합
  ├─ load_data(nb_1km=True, nb_3km=True, ..., n_20_nb=True)
  ├─ id 컬럼 기준 merge
  │  ├─ 결측치는 0으로 채우기
  │  └─ 순서 보정
  ├─ 최종 특성 수: 1000+ (원본 + 이웃 특성)
  └─ 형태: X_train (21,539 × 1000+), X_test (6,498 × 1000+)
  ↓
[5-2] 특성 스케일링 (선택적: do_scale=True)
  ├─ 수치형 특성만 스케일링
  ├─ RobustScaler() 사용 (이상치에 강함)
  │  ├─ (X - median) / IQR
  │  └─ 트리 기반 모델은 불필요, 신경망은 필수
  ├─ 범주형 특성은 유지
  └─ 주의: 훈련 데이터로 fit한 후 테스트에 transform
  ↓
[5-3] 메모리 최적화 (마지막)
  ├─ reduce_mem_usage()
  ├─ dtype 최적화
  └─ 메모리: 1000 특성 × 28,037행 → ~500MB
  ↓
출력: X_train, X_test, y_train (스택 준비)
```

### Phase 6: Stage 1 - 기초 모델 학습 (OOF 생성)

```
입력: X_train (21,539 × 1000+), y_train (21,539,), X_test (6,498 × 1000+)
  ↓
[6-1] 모델별 OOF 생성: get_oof()
  ├─ 모델 1: XGBoost
  │  ├─ 5-fold CV split
  │  │  ├─ Fold 1: train[0:4308] vs valid[4308:8616]
  │  │  ├─ Fold 2: train[0:4308, 8616:12924] vs valid[4308:8616]
  │  │  ├─ ...
  │  │  └─ Fold 5: train[0:17231] vs valid[17231:21539]
  │  ├─ 각 폴드마다:
  │  │  ├─ xgb.train(dtrain, dvalid, 500라운드)
  │  │  ├─ valid 세트 예측 → oof_train[valid_idx]
  │  │  ├─ cv_score = rmse(y_valid, pred)
  │  │  └─ 조기 종료로 최적 라운드 기록
  │  ├─ 전체 데이터 재학습 → test 예측
  │  └─ 반환: oof_train (21,539,), oof_test (6,498,), cv_score (float)
  │
  ├─ 모델 2: LightGBM
  │  ├─ 동일 5-fold 분할
  │  ├─ lgb.train() 사용
  │  └─ 반환: (21,539,), (6,498,), cv_score
  │
  ├─ 모델 3: CatBoost (범주형 변수 처리)
  ├─ 모델 4: Ridge (선형)
  ├─ 모델 5: Lasso (선형 + 정규화)
  ├─ 모델 6: ElasticNet (혼합 정규화)
  ├─ 모델 7: RandomForest
  └─ 모델 8+: 신경망, Keras 임베딩 등
  ↓
[6-2] OOF 결과 결합
  ├─ oof_train_list = [모델1_oof, 모델2_oof, ...]
  ├─ np.concatenate(oof_train_list, axis=1)
  │  └─ 형태: (21,539, 8) - 8개 모델
  ├─ pd.DataFrame()화
  └─ X_train_stage1 (21,539 × 8) 완성
  ↓
[6-3] Stage 1 결과 저장
  ├─ x_train_stage1.csv: 훈련 세트 OOF
  ├─ x_test_stage1.csv: 테스트 세트 예측
  └─ cv_score_stage1.csv: 각 모델의 CV 점수
  ↓
[6-4] 모델 성능 분석
  ├─ 평균 CV RMSE: 0.12~0.15 (로그 스케일)
  ├─ 모델별 성능 차이
  │  ├─ 예: XGBoost 0.123, LightGBM 0.125, Ridge 0.142
  │  └─ 앙상블 성능 = 다양성 + 평균 성능
  └─ 로그 공간에서 RMSE = 0.12 → 원본 스케일에서 약 12~13% 오차
  ↓
특징: OOF 예측이 훈련 데이터에 대한 "의견" 역할 - 메타 모델이 이를 학습
```

### Phase 7: 메타 특성 생성

```
입력: X_train_stage1 (21,539 × 8), X_test_stage1 (6,498 × 8)
  ↓
[7-1] 메타 특성 그대로 사용 (가장 간단)
  ├─ X_train_stage2 = X_train_stage1
  ├─ X_test_stage2 = X_test_stage1
  └─ 8개 기초 모델 OOF가 메타 입력이 됨
  ↓
[7-2] 추가 메타 특성 생성 (선택적)
  ├─ OOF 간 상호작용
  │  ├─ OOF1 * OOF2, OOF1 / OOF2, OOF1 - OOF2 등
  │  └─ 추가 특성으로 사용
  ├─ OOF의 통계
  │  ├─ mean(OOF1~OOF8)
  │  ├─ std(OOF1~OOF8)
  │  ├─ max, min
  │  └─ 이웃 OOF의 통계
  └─ 선택적: 복잡성 증가 vs 약간의 성능 향상
  ↓
출력: X_train_stage2, X_test_stage2 (21,539 × 8~20, 6,498 × 8~20)
```

### Phase 8: Stage 2 - 메타 모델 학습

```
입력: X_train_stage2 (21,539 × 8), y_train (21,539,), X_test_stage2 (6,498 × 8)
  ↓
[8-1] 메타 모델 선택
  ├─ 추천: XGBoost (단순하고 효과적)
  ├─ 대안: LightGBM, Neural Network, Weighted Average
  └─ 회피: 과도하게 복잡한 모델
  ↓
[8-2] 메타 모델 학습: get_oof()
  ├─ 입력: 8개 OOF 특성
  ├─ 5-fold CV:
  │  ├─ Fold 1: 기초 모델1~5 학습, 모델6 예측 사용
  │  │         → oof_train_meta[fold1]
  │  ├─ ...
  │  └─ Fold 5
  ├─ 전체 재학습 → test 예측
  └─ 반환: oof_final (21,539,), pred_final (6,498,), final_cv_score
  ↓
[8-3] 최종 성능
  ├─ 메타 모델 CV RMSE: 0.10~0.13 (로그 스케일)
  │  └─ Stage 1 평균 (0.12~0.15)보다 개선
  ├─ 원본 스케일에서 약 10~13% 오차
  └─ Kaggle 테스트 세트에서는 조금 더 높을 수 있음 (과적합)
  ↓
특징: Stage 1 OOF의 패턴을 메타 모델이 학습하여 앙상블 성능 최적화
```

### Phase 9: 최종 예측 및 제출

```
입력: pred_final (6,498,) - 로그 스케일 예측
  ↓
[9-1] 역변환
  ├─ np.expm1(pred_final) = exp(pred) - 1
  ├─ log 스케일 → 원본 가격 스케일 복원
  └─ 범위: 0 ~ 수천만 원
  ↓
[9-2] 결과 준비
  ├─ test.csv에서 id 추출
  ├─ pd.DataFrame({'id': ids, 'price': pred_final_exp})
  └─ 형태: (6,498 × 2)
  ↓
[9-3] CSV 제출
  ├─ submission.csv 저장
  ├─ 형식: id,price
  │  └─ 예:
  │     id,price
  │     1,300000
  │     2,450000
  │     ...
  └─ 파일 크기: ~50KB
  ↓
완료!
```

## 데이터 형태 변환 요약

```
Stage   형태 변화                           행 수        열 수      설명
────────────────────────────────────────────────────────────────────────
Raw     train.csv + test.csv              28,037        21        원본 데이터
        └─ 이상치 제거

Phase1  X_train_single + X_test_single    28,037      1000+       피처 엔지니어링

Phase2  + 이웃 특성 (1km, 3km, 5km, 5NN..) 28,037     1000+120    지리 특성 추가

Phase5  최종 통합 X_train, X_test         28,037      1000+       학습용 데이터
        ├─ X_train: 21,539 행
        └─ X_test:  6,498 행

Stage1  OOF 예측 (8개 모델)                21,539/6,498    8      기초 모델 예측
        ├─ x_train_stage1: (21,539, 8)
        └─ x_test_stage1:  (6,498, 8)

Stage2  메타 특성                         21,539/6,498   8-20     (동일, 선택적 확장)

Final   최종 예측                          6,498         1        로그 스케일
        ├─ pred_final (로그): (6,498,)
        └─ submission (원본): (6,498, 2) [id, price]
```

## 메모리 사용량 추정

```
데이터 단계               메모리 사용량      주요 요인
────────────────────────────────────────────────────
원본 데이터              ~25 MB           21개 특성 × 28K행
                         (reduce후 ~12MB)

Phase 1 (피처)           ~500 MB          1000+ 특성 × 28K행
(X_train + X_test)       (optimized)

Phase 2 (이웃)           ~600 MB          추가 120개 특성
(모든 이웃 파일)

Stage 1 OOF              ~50 MB           8 모델 × 28K행

메타 모델 학습           ~600 MB          Stage 1 입력 + 모델

피크 메모리              ~1.5 GB          동시 처리 시
(전체 과정)
```

## 키 포인트: 데이터 누수 방지

```
훈련 데이터에서만 계산해야 할 통계:
  ├─ 범주형 통계 (grade별 평균 가격): 훈련 데이터만 사용 ✓
  ├─ Scaler fit: 훈련 데이터만 fit ✓
  ├─ OOF: K-Fold로 각 샘플이 검증 데이터로 정확히 1회만 사용 ✓
  └─ 테스트 데이터에는 이웃이 있어도 훈련 데이터 통계 사용 ✓

회피할 것:
  ├─ 이웃 특성을 계산할 때 테스트 데이터 포함 ✗
  ├─ 전체 데이터(훈련+테스트)의 통계를 훈련에 사용 ✗
  └─ 테스트 데이터로 Scaler fit ✗
```
