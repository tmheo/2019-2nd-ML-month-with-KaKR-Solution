# Kaggle 주택 가격 예측 솔루션 - 아키텍처 코드맵

**프로젝트**: 2019 2nd ML month with KaKR - 주택 가격 예측
**생성일**: 2026-03-02
**언어**: 한국어 (ko)
**목적**: 다단계 스태킹 앙상블 기반 ML 솔루션의 아키텍처 시각화 및 문서화

## 문서 구성

### 1. [`overview.md`](./overview.md) - 아키텍처 개요 (13 KB)

**포함 내용**:
- 프로젝트 정보 및 기술 스택
- 전체 파이프라인 흐름 (ASCII 다이어그램)
- 다단계 스태킹 설계 패턴 상세 설명
- 지리적 특성 엔지니어링 개념
- 시스템 경계 정의

**대상 독자**: 프로젝트 개요를 파악하려는 개발자, 아키텍처 검토자

**사용 시기**:
- 처음 프로젝트를 접할 때
- 전체 파이프라인을 빠르게 이해하고 싶을 때

---

### 2. [`modules.md`](./modules.md) - 모듈 설명 (11 KB)

**포함 내용**:
- utils.py 전체 구조 (842줄)
- 6개 모델 래퍼 클래스 상세 설명 (SklearnWrapper, XgbWrapper, LgbmWrapper, CatWrapper, KerasWrapper, KerasEmbeddingWrapper)
- 18개 핵심 함수 및 메서드 문서화
- 공개 인터페이스 사용 예시

**상세 내용**:
- 클래스별 파라미터, 메서드, 반환값
- 함수별 입출력 형식 및 동작 원리
- 실제 사용 코드 예시

**대상 독자**: 구현 세부사항을 알아야 하는 개발자, 코드 리뷰어

**사용 시기**:
- 특정 함수 또는 클래스의 동작 방식을 알고 싶을 때
- 새로운 모델을 래퍼로 추가할 때
- 모듈 사용법을 배울 때

---

### 3. [`dependencies.md`](./dependencies.md) - 의존성 그래프 (12 KB)

**포함 내용**:
- 외부 패키지 의존성 (pandas, numpy, scikit-learn, XGBoost, LightGBM, CatBoost, Keras, TensorFlow 등)
- 내부 모듈 의존성 (노트북 ↔ utils.py 관계)
- 함수 호출 그래프 (get_oof, stacking, load_data 등)
- 데이터 파일 의존성 (입출력 파일 추적)
- 순환 의존성 분석 (없음 ✓)

**버전 호환성**:
- 권장 환경 명시 (Python 3.5+, pandas 0.23+ 등)
- TensorFlow 2.0+ 호환성 주의 사항

**대상 독자**: 환경 설정 담당자, 의존성 관리자

**사용 시기**:
- 환경을 설정할 때
- 특정 버전으로 업그레이드할 때
- 모듈 간 호출 관계를 파악할 때

---

### 4. [`entry-points.md`](./entry-points.md) - 진입점 및 실행 가이드 (15 KB)

**포함 내용**:
- 노트북 실행 순서 (4단계)
- 각 노트북의 상세 실행 가이드:
  - Step 1: EDA 및 피처 엔지니어링 (geo-data-eda-and-feature-engineering.ipynb)
  - Step 2: 지리적 이웃 정보 추출 (Generate Neighbor Info.ipynb)
  - Step 3: 이웃 통계 집계 (Generate Neighbor Stat.ipynb)
  - Step 4: 스태킹 앙상블 (Stacking Ensemble.ipynb)

**각 단계별 정보**:
- 입출력 파일 목록
- 주요 코드 구조 (의사코드)
- 생성되는 특성 설명
- 실행 팁 및 최적화 방법

**병렬 실행 가능성**:
- 병렬 가능한 단계 표시
- 순차 필수 단계 강조

**대상 독자**: 프로젝트를 처음 실행하려는 사용자

**사용 시기**:
- 처음 솔루션을 실행할 때
- 특정 단계를 다시 실행해야 할 때
- 실행 시간 및 메모리 요구사항을 알고 싶을 때

---

### 5. [`data-flow.md`](./data-flow.md) - 데이터 흐름 분석 (17 KB)

**포함 내용**:
- 전체 데이터 파이프라인 (9개 Phase 상세 분석)
- 각 Phase별 데이터 변환 과정
  - Phase 1: 데이터 로드 및 기본 전처리
  - Phase 2: 피처 엔지니어링 (원본 특성 생성)
  - Phase 3: 지리적 이웃 특성
  - Phase 4: 이웃 특성 통계
  - Phase 5: 데이터 통합 및 스케일링
  - Phase 6: Stage 1 기초 모델 학습
  - Phase 7: 메타 특성 생성
  - Phase 8: Stage 2 메타 모델 학습
  - Phase 9: 최종 예측 및 제출

**상세 정보**:
- 각 Phase의 입출력 형태
- 데이터 변환 알고리즘 및 공식
- 메모리 사용량 추정
- 복잡도 분석 (Big-O)

**데이터 누수 방지**:
- 훈련/테스트 데이터 분리 원칙
- 피해야 할 실수 목록

**데이터 형태 요약표**:
| Stage  | 형태 변화 | 행 수 | 열 수 | 설명 |
|--------|----------|-------|-------|------|
| Raw    | 원본     | 28K   | 21    | 원본 데이터 |
| Phase1 | 피처     | 28K   | 1000+ | 피처 엔지니어링 |
| ...    | ...      | ...   | ...   | ... |

**대상 독자**: 데이터 흐름을 깊이 있게 이해하려는 개발자, 데이터 엔지니어

**사용 시기**:
- 특정 변환 과정을 이해하고 싶을 때
- 메모리 최적화가 필요할 때
- 데이터 누수를 방지하려고 할 때

---

## 빠른 시작

### 초보자
1. [`overview.md`](./overview.md) - 전체 그림 이해 (5분)
2. [`entry-points.md`](./entry-points.md) - 실행 단계별 가이드 (10분)

### 개발자
1. [`modules.md`](./modules.md) - 코드 구조 이해 (15분)
2. [`dependencies.md`](./dependencies.md) - 환경 설정 (5분)

### 데이터 사이언티스트
1. [`data-flow.md`](./data-flow.md) - 데이터 변환 과정 (20분)
2. [`overview.md`](./overview.md) - 설계 패턴 이해 (5분)

### 아키텍처 검토자
1. [`overview.md`](./overview.md) - 전체 개요 (5분)
2. [`dependencies.md`](./dependencies.md) - 모듈 관계 (10분)
3. [`data-flow.md`](./data-flow.md) - 데이터 흐름 (15분)

---

## 문서 생성 통계

```
파일 구성:
├─ overview.md          13 KB   (시스템 아키텍처)
├─ modules.md           11 KB   (6개 클래스 + 18개 함수)
├─ dependencies.md      12 KB   (패키지 + 모듈 + 데이터 파일)
├─ entry-points.md      15 KB   (4단계 노트북)
└─ data-flow.md         17 KB   (9개 Phase 분석)

총계:
  - 파일 수: 6개 (README 포함)
  - 총 크기: ~68 KB
  - 총 단어: ~15,000 단어
  - 코드 예시: 40+ 개
  - ASCII 다이어그램: 8개
  - 표: 5개
```

---

## 프로젝트 구조

```
2019-2nd-ML-month-with-KaKR-Solution/
├─ notebook/
│  ├─ utils.py
│  ├─ geo-data-eda-and-feature-engineering.ipynb
│  ├─ Generate Neighbor Info.ipynb
│  ├─ Generate Neighbor Stat.ipynb
│  └─ Stacking Ensemble.ipynb
├─ input/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ neighbor_*km_stat.csv
│  ├─ nearest_*_neighbor_stat.csv
│  ├─ x_train_*.csv
│  └─ x_test_*.csv
└─ .moai/project/codemaps/ ← 현재 위치
   ├─ README.md
   ├─ overview.md
   ├─ modules.md
   ├─ dependencies.md
   ├─ entry-points.md
   └─ data-flow.md
```

---

## 핵심 설계 패턴

### 다단계 스태킹 (Multi-Stage Stacking)
- **Stage 1**: 6-10개 다양한 기초 모델이 각각 OOF 예측 생성
- **Stage 2**: 기초 모델 OOF를 새로운 특성으로 메타 모델 학습
- **성능 향상**: 모델 다양성 + 메타 학습으로 일반화 성능 향상

### 지리적 특성 엔지니어링
- **Haversine 거리**: 정확한 GPS 기반 거리 계산
- **반경 기반 집계**: 1km, 3km, 5km 반경 내 이웃의 특성 통계
- **K-NN 특성**: 가장 가까운 5, 10, 20개 이웃의 통계

### 데이터 누수 방지
- **K-Fold OOF**: 각 샘플이 검증 데이터로 정확히 1회만 사용
- **훈련 데이터만 학습**: 범주형 통계, Scaler fit 등은 훈련 데이터만 사용
- **테스트 데이터 미포함**: 이웃 특성 계산에 테스트 데이터 미포함

---

## 성능 메트릭

| 단계 | 모델 | CV RMSE (로그) | 원본 스케일 오차 |
|------|------|----------------|-----------------|
| Stage 1 | XGBoost | 0.123 | ~12.3% |
| Stage 1 | LightGBM | 0.125 | ~12.5% |
| Stage 1 | Ridge | 0.142 | ~14.2% |
| Stage 1 | 평균 | 0.130 | ~13% |
| Stage 2 | XGBoost (메타) | **0.110** | **~11%** |

*참고: 실제 성능은 데이터 및 하이퍼파라미터에 따라 변동*

---

## 주요 기술 스택

### 데이터 처리
- **pandas**: 데이터프레임 조작
- **numpy**: 수치 연산 및 벡터화

### 머신러닝
- **scikit-learn**: 전처리, 선형 모델, 앙상블
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 빠른 부스팅
- **CatBoost**: 범주형 변수 최적화

### 딥러닝
- **Keras/TensorFlow**: 신경망 모델

### 시각화
- **matplotlib/seaborn**: EDA 시각화

---

## 실행 시간 추정

| 단계 | 소요 시간 | GPU 필요 | 메모리 |
|------|----------|----------|--------|
| Step 1: EDA | 30-60분 | × | ~500MB |
| Step 2: 이웃 추출 | 10-20분 | × | ~200MB |
| Step 3: 이웃 통계 | 20-40분 | × | ~300MB |
| Step 4: 앙상블 | 2-4시간 | ○ | ~1.5GB |
| **총계** | **3-5시간** | ○ (권장) | **~1.5GB** |

*GPU 없을 경우 추가 2-3배 시간 소요*

---

## 문서 사용 팁

1. **검색**: 특정 함수명으로 modules.md 검색
2. **참조**: 호출 관계는 dependencies.md 참조
3. **실행**: entry-points.md의 코드 예시 복사
4. **이해**: data-flow.md의 Phase별 다이어그램으로 시각화

---

## 개선 사항 및 확장 가능성

### 개선 기회
- 신경망 모델 추가 (LSTM, Transformer)
- 특성 선택 알고리즘 (특성 중요도 기반)
- 하이퍼파라미터 자동 튜닝 (Grid Search, Bayesian Optimization)
- 앙상블 가중치 최적화 (검증 데이터 기반)

### 성능 향상 전략
- 더 많은 이웃 반경 추가 (0.5km, 2km, 10km)
- 시계열 특성 강화 (계절성, 트렌드)
- 외부 데이터 통합 (경제 지표, 교통 데이터)
- 특성 상호작용 확대

---

## 관련 자료

**원본 파일**:
- Kaggle Competition: [House Price Prediction](https://www.kaggle.com/c/kc-house-data-prediction)
- 데이터셋: King County, Washington 주택 거래 데이터

**라이브러리 공식 문서**:
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Keras Documentation](https://keras.io/api/)

---

**작성자**: MoAI Documentation Manager
**생성일**: 2026-03-02
**언어**: 한국어 (한국)
**버전**: 1.0.0
