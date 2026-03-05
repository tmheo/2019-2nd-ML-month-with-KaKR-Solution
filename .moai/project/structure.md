# 프로젝트 구조 가이드

## 디렉토리 트리

```
2019-2nd-ML-month-with-KaKR-Solution/
├── notebook/                                    # Jupyter Notebook 및 실행 코드
│   ├── geo-data-eda-and-feature-engineering.ipynb
│   ├── Generate Neighbor Info.ipynb
│   ├── Generate Neighbor Stat.ipynb
│   ├── Stacking Ensemble.ipynb
│   ├── utils.py                                # 공유 유틸리티 및 모델 래퍼
│   └── stacking_20190421093422.csv             # 최종 예측 결과
│
├── input/                                       # 데이터 디렉토리
│   ├── train.csv                               # 학습 데이터 (21,600 행)
│   ├── test.csv                                # 테스트 데이터 (6,900 행)
│   │
│   ├── neighbor_1km_stat.csv                   # 반경 1km 이웃 통계
│   ├── neighbor_3km_stat.csv                   # 반경 3km 이웃 통계
│   ├── neighbor_5km_stat.csv                   # 반경 5km 이웃 통계
│   │
│   ├── nearest_5_neighbor_stat.csv             # 5-최근접 이웃 통계
│   ├── nearest_10_neighbor_stat.csv            # 10-최근접 이웃 통계
│   ├── nearest_20_neighbor_stat.csv            # 20-최근접 이웃 통계
│   │
│   ├── x_train_single.csv                      # Stage 1 Out-of-Fold 예측 (학습)
│   ├── x_test_single.csv                       # Stage 1 기본 모델 예측 (테스트)
│   │
│   ├── x_train_stage2.csv                      # Stage 2 메타 특성 (학습)
│   ├── x_test_stage2.csv                       # Stage 2 메타 특성 (테스트)
│   │
│   ├── cv_score_*.csv                          # 교차 검증 점수
│   └── y_train.csv                             # 목표 변수 (판매 가격)
│
├── README.md                                    # 프로젝트 개요
├── .gitignore                                   # Git 무시 파일
└── .moai/                                       # MoAI 설정 (선택사항)
    └── project/                                # 프로젝트 문서
        ├── product.md                          # 제품 정의서
        ├── structure.md                        # 구조 가이드 (현재 파일)
        └── tech.md                             # 기술 스택 정보
```

## 디렉토리 및 파일 설명

### notebook/ 디렉토리
Jupyter Notebook과 Python 유틸리티를 포함하는 주요 작업 디렉토리입니다.

#### 주요 파일

**geo-data-eda-and-feature-engineering.ipynb** (41 셀)
- **목적**: 탐색적 데이터 분석(EDA) 및 특성 공학
- **주요 작업**:
  - 원본 데이터 로드 및 기본 통계 분석
  - 결측치(Missing Values) 처리
  - 이상치(Outliers) 탐지 및 제거
  - 날짜 특성 추출 (년도, 월, 계절 등)
  - 비율 특성 생성 (면적 비율, 방의 수 비율 등)
  - 지리적 특성 생성:
    - PCA를 이용한 좌표 압축
    - K-Means 클러스터링
    - 방위각(Bearing) 계산
  - Box-Cox 변환으로 왜도 정규화
  - 인코딩 (원-핫 인코딩, 레이블 인코딩)
- **출력**: 전처리된 학습/테스트 데이터

**Generate Neighbor Info.ipynb** (12 셀)
- **목적**: 지리적 이웃 정보 추출
- **주요 작업**:
  - Haversine 공식으로 구면 거리 계산
  - 반경별 이웃 추출:
    - 1km, 3km, 5km 반경 내 주택 검색
  - K-최근접 이웃:
    - 5개, 10개, 20개 최근접 이웃 식별
  - 이웃 정보를 CSV로 저장
- **입력**: 학습/테스트 데이터
- **출력**: neighbor_*km_stat.csv, nearest_*_neighbor_stat.csv

**Generate Neighbor Stat.ipynb** (21 셀)
- **목적**: 이웃 주택의 통계 정보 집계
- **주요 작업**:
  - 각 주택별 이웃의:
    - 가격 평균, 중앙값, 표준편차
    - 건축 년도 통계
    - 건축 면적 통계
  - 통계 정보를 메타 특성으로 변환
  - 학습/테스트 데이터에 병합
- **입력**: 이웃 정보 파일
- **출력**: 통계 포함된 특성 파일

**Stacking Ensemble.ipynb** (27 셀)
- **목적**: 앙상블 모델 구축 및 최종 예측
- **주요 작업**:
  - Stage 1: 기본 모델 학습
    - ElasticNet, Lasso, Ridge (선형 모델)
    - RandomForest, ExtraTrees (트리 기반)
    - XGBoost, LightGBM, CatBoost (부스팅)
    - Keras Dense 신경망
  - K-Fold Cross-Validation으로 Out-of-Fold 예측 생성
  - Stage 2: 메타 모델 학습
    - Stage 1 예측을 입력으로 사용
    - 최종 메타 모델로 예측 수행
  - 최종 앙상블 예측 생성
  - Kaggle 제출 형식으로 저장
- **입력**: 특성 공학된 학습/테스트 데이터
- **출력**: stacking_*.csv (최종 예측)

**utils.py** (842줄)
- **목적**: 모든 Notebook에서 공유하는 유틸리티 함수 및 모델 래퍼
- **핵심 함수**:
  - `reduce_mem_usage()`: 데이터프레임 메모리 최적화
  - `haversine_array()`: Haversine 거리 계산
  - `load_data()`: 포괄적 데이터 로딩 파이프라인
  - 모델 래퍼 클래스 (6가지):
    - `SklearnWrapper`: scikit-learn 모델
    - `XgbWrapper`: XGBoost 모델
    - `LgbmWrapper`: LightGBM 모델
    - `CatWrapper`: CatBoost 모델
    - `KerasWrapper`: Keras 신경망
    - `KerasEmbeddingWrapper`: Embedding 층 활용
  - `get_oof()`: K-Fold Out-of-Fold 예측 생성
  - `stacking()`: 다단계 앙상블 오케스트레이션
  - `rmse()`, `rmse_exp()`: 평가 메트릭
- **의존성**: pandas, numpy, scikit-learn, XGBoost, LightGBM, CatBoost, Keras/TensorFlow

**stacking_20190421093422.csv**
- **목적**: 최종 예측 결과
- **내용**: 테스트 데이터에 대한 최종 주택 가격 예측

### input/ 디렉토리
모든 데이터 파일을 포함하는 디렉토리입니다. Kaggle 경진대회에서 다운로드한 데이터와 중간 처리 결과를 저장합니다.

#### 데이터 파일 분류

**원본 데이터**
- `train.csv`: 학습 데이터 (Id, 79개 특성, SalePrice)
- `test.csv`: 테스트 데이터 (Id, 79개 특성, 가격 없음)

**지리적 이웃 특성**
- `neighbor_*km_stat.csv`: 반경 기반 이웃 통계
- `nearest_*_neighbor_stat.csv`: K-최근접 이웃 통계

**앙상블 중간 결과**
- `x_train_single.csv`: Stage 1 기본 모델들의 Out-of-Fold 예측
- `x_test_single.csv`: Stage 1 기본 모델들의 테스트 예측
- `x_train_stage2.csv`: Stage 2 학습용 메타 특성
- `x_test_stage2.csv`: Stage 2 테스트용 메타 특성

**평가 결과**
- `cv_score_*.csv`: 각 모델의 교차 검증 점수

## 모듈 구성

### 모듈 1: 데이터 처리 (Data Processing)
**파일**: `notebook/utils.py`, `notebook/geo-data-*.ipynb`
**책임**:
- 데이터 로딩 및 전처리
- 특성 공학 및 변환
- 메모리 최적화

### 모듈 2: 지리적 특성 (Geographic Features)
**파일**: `notebook/Generate Neighbor*.ipynb`, `notebook/utils.py`
**책임**:
- 거리 계산
- 이웃 추출
- 통계 집계

### 모듈 3: 모델 학습 (Model Training)
**파일**: `notebook/Stacking Ensemble.ipynb`, `notebook/utils.py`
**책임**:
- 기본 모델 학습
- Out-of-Fold 예측 생성
- 메타 모델 학습

### 모듈 4: 모델 평가 (Model Evaluation)
**파일**: `notebook/Stacking Ensemble.ipynb`, `notebook/utils.py`
**책임**:
- 교차 검증 수행
- RMSE 계산
- 모델 성능 비교

## 실행 순서

### 데이터 준비 단계

**1단계**: `Generate Neighbor Info.ipynb` 실행
- 학습/테스트 데이터에서 지리적 이웃 정보 추출
- 반경별 및 K-최근접 이웃 CSV 파일 생성
- 소요 시간: 약 10~20분

**2단계**: `Generate Neighbor Stat.ipynb` 실행
- 이웃 정보를 기반으로 통계 정보 계산
- 이웃 통계 CSV 파일 생성
- 소요 시간: 약 5~10분

### 특성 공학 단계

**3단계**: `geo-data-eda-and-feature-engineering.ipynb` 실행
- 원본 데이터 탐색 및 분석
- 다양한 특성 공학 적용
- 이웃 통계 특성 통합
- 최종 학습/테스트 데이터 생성
- 소요 시간: 약 30~40분

### 모델 학습 단계

**4단계**: `Stacking Ensemble.ipynb` 실행
- Stage 1 기본 모델 학습 및 Out-of-Fold 예측
- Stage 2 메타 특성 준비
- Stage 2 메타 모델 학습
- 최종 앙상블 예측 생성
- Kaggle 제출 파일 생성
- 소요 시간: 약 1~2시간 (GPU 사용 시 단축 가능)

## 데이터 흐름(Data Flow)

```
원본 데이터                          이웃 정보 추출
  ↓                                    ↓
train.csv, test.csv  ──→  neighbor_*.csv, nearest_*.csv
  ↓                         ↓
  └──────────────────────┬──────────────────────┘
                         ↓
                   이웃 통계 생성
                         ↓
               neighbor_*_stat.csv
                         ↓
                    ┌────┴────┐
                    ↓         ↓
                EDA 분석   특성 공학
                    ↓         ↓
                    └────┬────┘
                         ↓
              전처리된 학습/테스트 데이터
                         ↓
                   ┌────────────────────┐
                   ↓                    ↓
            Stage 1 기본 모델      10+ 모델 조합
                   ↓                    ↓
            Out-of-Fold 예측   ──→  메타 특성
                   ↓                    ↓
                   └────────┬───────────┘
                            ↓
                     Stage 2 메타 모델
                            ↓
                     최종 앙상블 예측
                            ↓
                     stacking_*.csv
```

## 의존성 및 파일 관계

```
utils.py (핵심 모듈)
  ├─ 다모든 Notebook에서 import됨
  ├─ 모델 래퍼 클래스 제공
  └─ 공유 함수 제공

Generate Neighbor Info.ipynb (1단계)
  ├─ input/train.csv, test.csv 읽기
  ├─ utils.py의 haversine_array() 함수 사용
  └─ output: neighbor_*.csv, nearest_*.csv 생성

Generate Neighbor Stat.ipynb (2단계)
  ├─ input/neighbor_*.csv, nearest_*.csv 읽기
  └─ output: 통계 특성 파일 생성

geo-data-eda-and-feature-engineering.ipynb (3단계)
  ├─ input/train.csv, test.csv 읽기
  ├─ input/neighbor_*_stat.csv 읽기
  ├─ utils.py의 reduce_mem_usage() 등 함수 사용
  └─ output: 특성 공학된 데이터 생성

Stacking Ensemble.ipynb (4단계)
  ├─ 3단계의 출력 데이터 읽기
  ├─ utils.py의 모든 래퍼 클래스 및 함수 사용
  ├─ utils.py의 get_oof(), stacking() 함수 사용
  └─ output: 최종 예측 생성
```

## 핵심 파일 위치

| 기능 | 파일 | 위치 |
|------|------|------|
| 거리 계산 | haversine_array() | notebook/utils.py (L250~280) |
| 메모리 최적화 | reduce_mem_usage() | notebook/utils.py (L78~120) |
| 모델 래퍼 | SklearnWrapper, XgbWrapper 등 | notebook/utils.py (L350~500) |
| OOF 생성 | get_oof() | notebook/utils.py (L550~650) |
| 앙상블 | stacking() | notebook/utils.py (L700~800) |
| EDA | 데이터 탐색 | notebook/geo-data-*.ipynb |
| 이웃 추출 | 거리 기반 검색 | notebook/Generate Neighbor Info.ipynb |
| 이웃 통계 | 집계 및 병합 | notebook/Generate Neighbor Stat.ipynb |
| 최종 예측 | 앙상블 조합 | notebook/Stacking Ensemble.ipynb |

## 특성 크기 변화

| 단계 | 특성 수 | 설명 |
|------|--------|------|
| 원본 데이터 | 79개 | Kaggle 제공 특성 |
| EDA 후 | ~100개 | 날짜, 비율, 범주 특성 추가 |
| 지리 특성 추가 | ~130개 | PCA, K-Means, 방위각 |
| 이웃 통계 추가 | ~200개 | 반경/K-최근접 이웃 통계 |
| 최종 특성 | 200~230개 | 인코딩 및 정규화 후 |

## 메모리 효율성

| 데이터셋 | 원본 크기 | 최적화 후 | 감소율 |
|---------|---------|----------|--------|
| train.csv | ~50MB | ~15MB | 70% |
| test.csv | ~15MB | ~5MB | 67% |
| 전체 특성 | ~200MB | ~60MB | 70% |

`reduce_mem_usage()` 함수는 자동으로 정수를 int8/int16/int32로, 부동소수점을 float32로 변환하여 메모리를 최적화합니다.

---

**마지막 업데이트**: 2019년 4월 21일
