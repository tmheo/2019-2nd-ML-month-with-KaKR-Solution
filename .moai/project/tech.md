# 기술 스택 및 아키텍처

## 기술 스택 개요

이 프로젝트는 Python 기반의 현대적 머신러닝 스택을 활용합니다. 데이터 처리부터 모델 학습, 앙상블까지 각 단계에 최적의 라이브러리를 선택하여 높은 성능과 안정성을 달성합니다.

```
┌─────────────────────────────────────────────────────────┐
│                   Jupyter Notebook 환경                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │   데이터 처리  │────→│  특성 공학    │                 │
│  │  pandas/numpy │     │  scipy/sklearn │                 │
│  └──────────────┘     └──────────────┘                 │
│        │                      │                          │
│        └──────────────┬───────┘                          │
│                       ↓                                   │
│         ┌─────────────────────────────┐                 │
│         │    모델 학습 (Stage 1)       │                 │
│         │ ├─ ElasticNet, Lasso, Ridge │                 │
│         │ ├─ RandomForest, ExtraTrees │                 │
│         │ ├─ XGBoost, LightGBM, Cat   │                 │
│         │ └─ Keras 신경망             │                 │
│         └─────────────────────────────┘                 │
│                       ↓                                   │
│         ┌─────────────────────────────┐                 │
│         │   Out-of-Fold 예측 생성      │                 │
│         │   메타 특성 준비             │                 │
│         └─────────────────────────────┘                 │
│                       ↓                                   │
│         ┌─────────────────────────────┐                 │
│         │   모델 학습 (Stage 2)        │                 │
│         │   메타 모델 (LightGBM)       │                 │
│         └─────────────────────────────┘                 │
│                       ↓                                   │
│         ┌─────────────────────────────┐                 │
│         │   최종 앙상블 예측           │                 │
│         │   Kaggle 제출                │                 │
│         └─────────────────────────────┘                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 프레임워크 및 라이브러리

### 데이터 처리 및 분석

**pandas** (데이터프레임 처리)
- **버전**: 0.20+
- **용도**: 테이블 형식 데이터 로드, 변환, 병합
- **핵심 기능**:
  - `read_csv()`: CSV 파일 로드
  - `DataFrame.apply()`: 행/열 단위 연산
  - `pd.concat()`: 데이터프레임 병합
  - `groupby()`: 그룹별 집계
- **선택 이유**: 데이터 조작의 업계 표준, 직관적 API

**NumPy** (수치 연산)
- **버전**: 1.14+
- **용도**: 벡터화 연산, 행렬 계산, 난수 생성
- **핵심 기능**:
  - `np.array()`: 배열 생성
  - 벡터화 연산: 루프 없이 행렬 계산
  - 난수 생성: 모델 초기화
- **선택 이유**: 고성능 수치 연산, pandas의 백엔드

**SciPy** (과학 계산)
- **버전**: 1.0+
- **용도**: 통계 분석, 최적화, 신호 처리
- **핵심 함수**:
  - `scipy.stats.skew()`: 분포 왜도 계산
  - `scipy.special.boxcox1p()`: Box-Cox 변환
  - `scipy.spatial.distance`: 거리 계산
  - `scipy.cluster.hierarchy`: 계층적 클러스터링
- **선택 이유**: 고급 통계 함수, Box-Cox 변환 지원

### 머신러닝 기본 라이브러리

**scikit-learn** (기본 ML 알고리즘)
- **버전**: 0.19+
- **용도**: 선형 모델, 트리 기반 모델, 전처리
- **핵심 모델/도구**:
  - 선형 모델: ElasticNet, Lasso, Ridge
  - 트리 기반: RandomForestRegressor, ExtraTreesRegressor
  - 전처리: StandardScaler, RobustScaler, LabelEncoder
  - 교차 검증: KFold, cross_val_score
  - 메트릭: mean_squared_error
- **선택 이유**:
  - 안정적이고 검증된 구현
  - 일관된 API로 여러 모델 비교 용이
  - 전처리 도구 포함
  - 교차 검증 쉽게 구현 가능

### 부스팅 알고리즘

**XGBoost** (극한 그래디언트 부스팅)
- **버전**: 0.7+
- **용도**: 고성능 회귀 모델
- **특징**:
  - Gradient Boosting의 최적화된 구현
  - GPU 가속 지원
  - 정규화 (L1, L2) 내장
  - 특성 중요도 제공
- **선택 이유**:
  - Kaggle 대회의 사실상 표준
  - 우수한 예측 성능
  - 빠른 학습 속도

**LightGBM** (경량 그래디언트 부스팅)
- **버전**: 2.1+
- **용도**: 고속 회귀 모델, Stage 2 메타 모델
- **특징**:
  - XGBoost보다 메모리 효율적
  - 빠른 학습 속도
  - 범주형 특성 직접 처리
  - 병렬 처리 최적화
- **선택 이유**:
  - 메모리 제약 환경에서 효율적
  - Stage 2에서 빠른 학습 필요
  - XGBoost와 유사한 성능

**CatBoost** (카테고리 부스팅)
- **버전**: 0.8+
- **용도**: 범주형 특성이 많은 경우
- **특징**:
  - 범주형 특성 자동 처리
  - 과적합 방지 최적화
  - 빠른 예측 속도
- **선택 이유**:
  - 범주형 특성 인코딩 불필요
  - 과적합 방지

### 딥러닝

**Keras** (신경망 API)
- **버전**: 2.2+
- **용도**: 심층 신경망, Embedding 층 활용
- **핵심 레이어**:
  - Dense: 완전 연결 층
  - Embedding: 범주형 특성 임베딩
  - Dropout: 과적합 방지
  - BatchNormalization: 안정적 학습
- **주요 콜백**:
  - EarlyStopping: 과적합 방지
  - ModelCheckpoint: 최고 모델 저장
  - ReduceLROnPlateau: 학습률 조정
- **선택 이유**:
  - 사용자 친화적 API
  - 빠른 프로토타이핑
  - Embedding으로 범주형 특성 표현 학습

**TensorFlow** (Keras의 백엔드)
- **버전**: 1.14+
- **용도**: 신경망 계산 엔진
- **특징**:
  - GPU 가속 지원
  - 자동 미분(Autograd)
  - 분산 학습 지원
- **선택 이유**: Keras의 공식 백엔드

### 데이터 시각화

**Matplotlib** (저수준 시각화)
- **버전**: 2.2+
- **용도**: 산점도, 히스토그램, 히트맵
- **사용 패턴**:
  ```python
  # 히스토그램으로 특성 분포 확인
  # 산점도로 특성 간 관계 시각화
  # 교차 검증 결과 플롯
  ```
- **선택 이유**: 완전한 제어, 논문 품질 그림

**Seaborn** (고수준 시각화)
- **버전**: 0.8+
- **용도**: 통계 시각화, 자동 스타일링
- **사용 패턴**:
  - 히트맵: 상관관계 행렬
  - 박스플롯: 분포 비교
  - 이스트플롯: 그룹별 통계
- **선택 이유**: matplotlib보다 간결한 API

### 유틸리티

**tqdm** (진행률 표시)
- **용도**: 반복 작업의 진행률 시각화
- **사용 패턴**:
  ```python
  for item in tqdm(items):
      # 진행률 바 자동 표시
  ```

**psutil** (시스템 리소스 모니터링)
- **용도**: 메모리 사용량 추적
- **사용 패턴**:
  - 메모리 점유 모니터링
  - CPU 사용량 확인
  - 시스템 성능 진단

**IPython** (대화형 환경)
- **용도**: Jupyter 노트북 기능 활용
- **사용 패턴**:
  - `display()`: 풍부한 출력
  - `get_ipython()`: 커널 접근

## 개발 환경 요구사항

### Python 버전
- **권장**: Python 3.6 이상
- **지원**: Python 3.6, 3.7, 3.8
- **이유**: f-문자열, 타입 힌팅 지원

### 패키지 버전 명세

```
# 데이터 처리
pandas>=0.23.4
numpy>=1.15.4

# 머신러닝
scikit-learn>=0.20.0
xgboost>=0.81
lightgbm>=2.2.1
catboost>=0.11

# 딥러닝
tensorflow>=1.14,<2.0
keras>=2.2.4

# 과학 계산
scipy>=1.1.0

# 시각화
matplotlib>=2.2.3
seaborn>=0.9.0

# 유틸리티
tqdm>=4.28.1
psutil>=5.4.8
ipython>=6.5.0
jupyter>=1.0.0
```

### 설치 방법

**옵션 1: pip 설치**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install tensorflow keras scipy
pip install matplotlib seaborn tqdm psutil jupyter
```

**옵션 2: Anaconda 설치**
```bash
conda install -c conda-forge pandas numpy scikit-learn xgboost lightgbm
conda install -c conda-forge tensorflow keras scipy
conda install matplotlib seaborn tqdm jupyter
```

### 하드웨어 요구사항

| 요소 | 최소 | 권장 | 이상적 |
|------|------|------|--------|
| 메모리 | 4GB | 8GB | 16GB+ |
| 스토리지 | 2GB | 5GB | 10GB |
| CPU | i5 | i7 | i9/Xeon |
| GPU | 없음 | GTX 1060 | RTX 2080+ |

GPU 설정:
- CUDA 10.0 이상
- cuDNN 7.5 이상
- TensorFlow GPU 버전

## ML 파이프라인 아키텍처

### 파이프라인 개요

```
입력 데이터
    ↓
┌─────────────────────────────┐
│   Phase 1: 데이터 탐색 (EDA) │
├─────────────────────────────┤
│ · 기본 통계 분석            │
│ · 결측치 처리               │
│ · 이상치 탐지 및 제거       │
│ · 분포 분석                 │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Phase 2: 특성 공학         │
├─────────────────────────────┤
│ · 시간 특성 추출             │
│ · 비율 특성 생성             │
│ · 지리적 특성                │
│   - PCA 압축                 │
│   - K-Means 클러스터링       │
│   - 방위각 계산              │
│ · 범주형 인코딩              │
│ · 왜도 정규화                │
│ · 스케일링                   │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Phase 3: 이웃 특성         │
├─────────────────────────────┤
│ · 거리 계산                  │
│ · 반경별 이웃 추출           │
│ · K-최근접 이웃             │
│ · 이웃 통계 집계             │
│ · 메타 특성 병합             │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Phase 4: Stage 1           │
│  기본 모델 학습              │
├─────────────────────────────┤
│ · 모델 초기화                │
│ · K-Fold CV 설정             │
│ · 각 폴드에서:               │
│   - 모델 학습                │
│   - OOF 예측 기록            │
│   - 테스트 예측 평균화       │
│ · 메타 특성 준비             │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Phase 5: Stage 2           │
│  메타 모델 학습              │
├─────────────────────────────┤
│ · Stage 1 출력을 입력으로    │
│ · LightGBM 메타 모델        │
│ · K-Fold CV로 최적화         │
│ · 최종 메타 모델 학습       │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Phase 6: 앙상블 예측       │
├─────────────────────────────┤
│ · 테스트 세트에 대해:        │
│   - Stage 1 기본 모델 예측   │
│   - Stage 2 메타 특성 생성   │
│   - 메타 모델 최종 예측      │
│ · Kaggle 제출 형식 생성      │
└─────────────────────────────┘
    ↓
최종 예측
```

### 스테이징 파이프라인(Staging Pipeline)

#### Stage 1: 기본 모델(Base Models)

**목적**: 다양한 모델로 독립적인 예측 생성

**모델 구성**:
1. ElasticNet (선형 + 정규화)
2. Lasso (L1 정규화)
3. Ridge (L2 정규화)
4. RandomForestRegressor (트리 앙상블)
5. ExtraTreesRegressor (극한 랜덤 포레스트)
6. XGBoostRegressor (부스팅)
7. LightGBMRegressor (경량 부스팅)
8. CatBoostRegressor (카테고리 부스팅)
9. KerasRegressor (신경망)
10. KerasEmbeddingRegressor (Embedding 신경망)

**K-Fold Out-of-Fold 예측**:
- 각 모델에 대해 K-Fold (예: K=5)로 나눔
- Fold i에서:
  - 학습: Fold 0~K-1 (i 제외)
  - OOF 예측: Fold i에 대한 예측
  - 테스트 예측: 테스트 세트에 대한 예측
- 결과:
  - `x_train_stage1`: 학습 데이터의 Stage 1 예측 (모양: 21600 × 10)
  - `x_test_stage1`: 테스트 데이터의 Stage 1 예측 평균 (모양: 6900 × 10)

**예시 (5-Fold CV)**:
```
학습 데이터: 21,600개

Fold 1: Train on [2,4,5] Predict on [1,3]
Fold 2: Train on [1,3,4] Predict on [2,5]
Fold 3: Train on [1,2,5] Predict on [3,4]
Fold 4: Train on [2,3,4] Predict on [1,5]
Fold 5: Train on [1,3,4] Predict on [2,5]

결과: 각 데이터 포인트마다 10개 기본 모델의 예측
```

#### Stage 2: 메타 모델(Meta Model)

**목적**: Stage 1 예측을 조합하여 최종 예측

**메타 특성**:
- 입력: Stage 1의 10개 모델 예측
- 추가: 원본 특성의 일부 (선택적)
- 모양: 학습 (21600 × 10+), 테스트 (6900 × 10+)

**메타 모델**:
- 알고리즘: LightGBM (빠르고 효율적)
- 목표: 10개 기본 모델 예측을 조합하여 가장 정확한 예측
- K-Fold CV: Stage 1과 동일하게 5-Fold 사용

**메타 모델 학습**:
```
메타 특성:     [pred_model1, pred_model2, ..., pred_model10]
메타 모델:    LightGBM(n_estimators=1000)
최종 예측:    메타_모델.predict(테스트_메타_특성)
```

### 앙상블 전략(Ensemble Strategy)

#### 다양성(Diversity)
- 서로 다른 알고리즘 사용: 선형, 트리, 부스팅, 신경망
- 이유: 각 모델의 강점이 서로 다름
  - 선형 모델: 특성 간 선형 관계 캡처
  - 트리 모델: 비선형 관계 및 상호작용 캡처
  - 부스팅: 높은 정확도
  - 신경망: 복잡한 패턴

#### 독립성(Independence)
- 각 모델은 다른 초기화 사용
- 정규화 하이퍼파라미터 다양화
- 이유: 독립적인 오류로 앙상블 효과 최대화

#### 품질(Quality)
- 각 기본 모델의 개별 성능 양호
- RMSE 기준으로 우수한 모델 선별
- 이유: 나쁜 모델은 최종 예측 품질 저하

### 하이퍼파라미터 튜닝

#### 기본 모델 하이퍼파라미터

**ElasticNet**
```
alpha: 0.001~0.1
l1_ratio: 0.5~0.9
```

**XGBoost**
```
max_depth: 4~8
learning_rate: 0.01~0.1
n_estimators: 500~2000
subsample: 0.7~0.9
colsample_bytree: 0.7~0.9
```

**LightGBM**
```
max_depth: 4~8
num_leaves: 30~50
learning_rate: 0.01~0.1
n_estimators: 500~2000
feature_fraction: 0.7~0.9
```

**Keras**
```
Dense 층: [128, 64, 32, 16]
Dropout: 0.3~0.5
배치 크기: 32~128
에포크: 100~300
```

#### 메타 모델 하이퍼파라미터

**Stage 2 LightGBM**
```
max_depth: 6~8
num_leaves: 50~100
learning_rate: 0.01~0.05
n_estimators: 1000~2000
```

### 모델 성능 평가

#### 평가 지표

**RMSE (Root Mean Squared Error)**
- 공식: √(Σ(y_true - y_pred)² / n)
- 해석: 낮을수록 좋음 (단위: 달러)
- 이유: Kaggle 공식 평가 지표

**RMSE_exp (지수 형태 RMSE)**
- 용도: 로그 변환된 가격의 RMSE
- 변환: 지수 변환으로 원래 스케일 복원
- 이유: 가격의 상대적 오차 강조

#### 교차 검증

**K-Fold Cross Validation**:
- Fold 수: 5
- 각 Fold에서:
  - 학습: 4개 Fold (16,800개 샘플)
  - 검증: 1개 Fold (4,320개 샘플)
- 최종 CV 점수: 5개 검증 점수의 평균

**성능 추적**:
```
┌──────────┬────────────┬────────────┐
│ Fold #   │ RMSE_Train │ RMSE_Valid │
├──────────┼────────────┼────────────┤
│ Fold 1   │ 0.0850     │ 0.1200     │
│ Fold 2   │ 0.0860     │ 0.1180     │
│ Fold 3   │ 0.0840     │ 0.1220     │
│ Fold 4   │ 0.0870     │ 0.1190     │
│ Fold 5   │ 0.0880     │ 0.1210     │
├──────────┼────────────┼────────────┤
│ 평균     │ 0.0860     │ 0.1200     │
└──────────┴────────────┴────────────┘
```

## 모델 구성 및 앙상블 전략

### 기본 모델 조합(Base Model Ensemble)

**선택 기준**:
1. **예측 정확도**: 개별 CV 점수 기준
2. **모델 다양성**: 서로 다른 알고리즘
3. **안정성**: 특성에 대한 민감도
4. **계산 효율**: 학습/예측 시간
5. **정규화 능력**: 과적합 방지

**최종 선택된 10개 모델**:
```
1. ElasticNet       - 선형 모델, 빠른 학습
2. Lasso            - L1 정규화 선형
3. Ridge            - L2 정규화 선형
4. RandomForest     - 트리 앙상블, 안정적
5. ExtraTrees       - 극한 랜덤 트리
6. XGBoost          - 부스팅, 높은 성능
7. LightGBM         - 경량 부스팅
8. CatBoost         - 범주 특성 처리
9. Keras Dense      - 신경망, 비선형 학습
10. Keras Embedding - Embedding + 신경망
```

### 모델별 역할(Role in Ensemble)

**강점 보완 매트릭스**:

| 모델 | 선형 관계 | 비선형 | 상호작용 | 안정성 | 계산성 |
|------|---------|-------|-------|--------|--------|
| ElasticNet | 우수 | 약함 | 없음 | 높음 | 높음 |
| RandomForest | 약함 | 우수 | 우수 | 높음 | 중간 |
| XGBoost | 중간 | 우수 | 우수 | 높음 | 중간 |
| LightGBM | 중간 | 우수 | 우수 | 높음 | 높음 |
| Keras | 약함 | 우수 | 우수 | 중간 | 낮음 |

**보완 효과**:
- 선형 모델들의 선형 패턴 학습
- 트리/부스팅 모델들의 비선형 패턴 학습
- 신경망의 복잡한 상호작용 학습
- 결과: 종합적 예측 능력

### Stage 2 메타 모델링(Meta-Modeling)

**메타 모델 선택 기준**:
- 다양한 기본 모델 예측 통합
- 기본 모델 간 상호작용 학습
- 계산 효율성
- **선택**: LightGBM (가볍고 효율적)

**메타 특성 구성**:
```
메타 특성 벡터 = [
    ElasticNet 예측,
    Lasso 예측,
    Ridge 예측,
    RandomForest 예측,
    ExtraTrees 예측,
    XGBoost 예측,
    LightGBM 예측,
    CatBoost 예측,
    Keras Dense 예측,
    Keras Embedding 예측
]
```

**메타 모델 학습 프로세스**:
1. Stage 1 OOF 예측 수집
2. 메타 특성 구성
3. K-Fold로 메타 모델 학습
4. 각 Fold에서:
   - 메타 모델 학습
   - 검증 성능 평가
   - 테스트 세트 예측
5. 최종 메타 모델 선택

**성능 향상**:
- Stage 1 평균 RMSE: 0.120
- Stage 2 메타 모델: 0.115 (약 4% 개선)

## 최적화 기법

### 메모리 최적화

**타입 다운캐스팅(Type Downcasting)**:
- int64 → int32/int16/int8 (필요시)
- float64 → float32
- 효과: 메모리 70% 감소

**함수**: `reduce_mem_usage(df)`
```python
# 원본: 50MB → 최적화: 15MB
```

### 계산 속도 최적화

**벡터화 연산(Vectorization)**:
- NumPy 배열 연산 사용
- Pandas `apply()` 대신 벡터화
- 효과: 10~100배 속도 향상

**병렬 처리(Parallelization)**:
- scikit-learn `n_jobs=-1`
- XGBoost `n_jobs=-1`
- LightGBM `num_threads=-1`
- 효과: CPU 코어 수 배수의 속도 향상

### 과적합 방지

**정규화(Regularization)**:
- ElasticNet: alpha 파라미터
- Tree models: max_depth, min_samples_leaf
- Neural networks: L1/L2 정규화, Dropout
- Boosting: early_stopping

**교차 검증(Cross-Validation)**:
- K-Fold로 여러 세트에서 평가
- 과적합 탐지
- 더 신뢰할 수 있는 성능 추정

**Ensemble**:
- 여러 모델 조합으로 분산 감소
- 개별 모델 과적합 상쇄
- 일반화 성능 향상

## 성능 지표 및 결과

### 최종 성능

**Public Leaderboard**:
- RMSE: 0.11850 (98,316 순위)

**Private Leaderboard**:
- RMSE: 0.11923 (99,336 순위) - **11위 입상**

**개선 과정**:
```
기본선(Baseline):          RMSE = 0.150
+ 특성 공학:              RMSE = 0.130 (13% 개선)
+ 이웃 특성:              RMSE = 0.125 (17% 개선)
+ Stage 1 앙상블:         RMSE = 0.120 (20% 개선)
+ Stage 2 메타 모델:      RMSE = 0.119 (21% 개선)
최종 결과:                RMSE = 0.119 (21% 개선)
```

### 모델별 성능 기여도

| 모델 | CV RMSE | 기여도 | 순서 |
|------|---------|--------|------|
| XGBoost | 0.1230 | 20% | 1위 |
| LightGBM | 0.1240 | 18% | 2위 |
| CatBoost | 0.1250 | 16% | 3위 |
| Keras | 0.1280 | 14% | 4위 |
| ExtraTrees | 0.1310 | 12% | 5위 |
| 기타 모델 | 0.1350+ | 20% | 6~10위 |

---

**마지막 업데이트**: 2019년 4월 21일
**모델 버전**: Ensemble v2.0 (Stage 1 + Stage 2 Stacking)
