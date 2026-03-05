# SPEC-CURRICULUM-001: ML 기초 교육 커리큘럼 (압축판)

## 개요

| 항목 | 내용 |
|------|------|
| SPEC ID | SPEC-CURRICULUM-001 |
| 제목 | 데브옵스 엔지니어를 위한 ML 기초 교육 커리큘럼 |
| 대상 | 데브옵스 엔지니어 (신입 ~ 5-6년차), ML 배경지식 없음 |
| 교육 기반 | 2019 2nd ML month with KaKR - 주택 가격 예측 프로젝트 |
| 총 교육 시간 | 약 10시간 (5회차 x 2시간) |

---

## 커리큘럼 구조

### 1회차: ML 개론 + EDA (2시간)

**목표**: ML이 무엇인지 이해하고, 데이터를 탐색하는 방법을 배운다.

#### 이론 (30분)
- **ML이란?**: 규칙 기반 vs 데이터 기반 학습
  - DevOps 비유: if-else 알람 규칙 vs 과거 장애 데이터로 이상 탐지 모델 학습
- **문제 유형**: 분류 / 회귀 / 군집화 → 이 프로젝트는 **회귀** (집값 예측)
- **평가 지표**: RMSE - 예측값과 실제값의 차이 측정
- **이상치**: 정상 범위를 벗어난 데이터 → 제거 필요

#### 실습 (60분)
- 환경 설정: `uv sync`, Jupyter 커널 등록, 노트북 실행
- `train.csv` 로드 및 탐색: `df.shape`, `df.head()`, `df.describe()`
- 시각화 실습 (`geo-data-eda-and-feature-engineering.ipynb` cell-8 ~ cell-16):
  - 히스토그램: 데이터 분포 확인
  - 산점도: 변수 간 관계 확인
  - 박스플롯: zipcode별 가격 분포
  - 위도/경도 기반 지도 시각화

#### 참조 코드
- `notebook/geo-data-eda-and-feature-engineering.ipynb`: cell-4 ~ cell-16

---

### 2회차: 피처 엔지니어링 (2시간)

**목표**: 원본 데이터에서 새로운 특성을 만들어 모델 성능을 높이는 방법을 배운다.

#### 이론 (20분)
- **피처 엔지니어링이란?**: 원본 데이터를 가공하여 모델이 더 잘 학습하도록 돕는 과정
  - DevOps 비유: 로그에서 "에러율", "p99 응답시간" 같은 파생 메트릭을 만드는 것
- **주요 기법**: 로그 변환, 비율 피처, 불리언 피처, 카테고리 인코딩(One-Hot)

#### 실습 (70분)
- `load_data()` 함수 워크스루 (utils.py):
  - 로그 변환: `np.log1p(price)` → 왜도 보정
  - 시간 피처: `yr_sold - yr_built` (건축 후 경과 년수)
  - 비율 피처: `bathrooms / bedrooms`, `sqft_living / floors`
  - 불리언 피처: `has_basement`, `is_renovated`
  - 상호작용: `sqft_living * grade`
- Zipcode 분해 실습 (cell-17 ~ cell-28):
  - 5자리 우편번호를 여러 방식으로 쪼개 새로운 피처 생성
  - 각 피처를 지도에 시각화하여 패턴 확인
  - CV Score 비교: 피처 추가 전 117,117 → 추가 후 116,084

#### 참조 코드
- `notebook/geo-data-eda-and-feature-engineering.ipynb`: cell-17 ~ cell-33
- `notebook/utils.py`: `load_data()` (line 563-814)

---

### 3회차: 지리 데이터 피처 엔지니어링 (2시간)

**목표**: 위도/경도를 활용한 피처 엔지니어링과 비지도학습 기초를 배운다.

#### 이론 (20분)
- **PCA**: 좌표 데이터를 변환하여 새로운 피처 생성
- **K-Means 클러스터링**: 가까운 집들을 그룹으로 묶어 "동네" 피처 생성
  - K 결정: Elbow Method vs CV Score 기반 선택
- **Haversine 거리**: 지구 곡률을 고려한 두 좌표 간 거리 계산
  - DevOps 비유: CDN에서 가장 가까운 엣지 서버를 찾는 것과 유사

#### 실습 (70분)
- PCA 변환 (cell-36 ~ cell-42):
  - `PCA(n_components=2).fit_transform(coord)` → 시각화 → CV Score 개선 확인
- K-Means 클러스터링 (cell-43 ~ cell-62):
  - Elbow Method로 K 탐색 → CV Score 기반 K=32 선택
  - 클러스터 시각화, Feature Importance에서 중요 클러스터 확인
- Haversine 거리 (cell-63 ~ cell-71):
  - 특정 집에서 모든 집까지 거리 계산
  - 반경 5km 이웃 시각화
- 이웃 통계 피처: `Generate Neighbor Info/Stat` 노트북 개념 설명

#### 참조 코드
- `notebook/geo-data-eda-and-feature-engineering.ipynb`: cell-34 ~ cell-72
- `notebook/utils.py`: `haversine_array()`, `bearing_array()`

---

### 4회차: 모델 학습과 교차 검증 (2시간)

**목표**: ML 모델을 학습시키고 교차 검증으로 성능을 평가하는 방법을 배운다.

#### 이론 (30분)
- **모델 종류** (간단 비교):
  - 선형 모델 (Ridge): 단순, 빠름, 해석 가능
  - 트리 기반 (RandomForest): 비선형 관계 포착
  - 부스팅 (LightGBM): 현업에서 가장 많이 사용, 높은 성능
- **과적합 vs 과소적합**:
  - DevOps 비유: 알람이 너무 민감 (과적합) vs 너무 둔감 (과소적합)
- **K-Fold 교차 검증**: 데이터를 K등분하여 K번 학습/검증 반복 → 신뢰할 수 있는 성능 측정
- **하이퍼파라미터**: `learning_rate`, `max_depth`, `num_leaves` 등
  - DevOps 비유: 커널 파라미터, JVM 옵션 같은 시스템 설정값

#### 실습 (60분)
- LightGBM으로 모델 학습 (cell-11):
  - `get_oof_lgb()` 함수로 5-Fold CV 실행
  - CV Score 확인, Feature Importance 시각화
- 하이퍼파라미터 실험:
  - `num_leaves` 변경 (7 vs 15 vs 63) → CV Score 변화 관찰
  - `learning_rate` 변경 → 학습 속도와 성능의 트레이드오프
- `utils.py` 모델 래퍼 패턴 간단 설명:
  - 다양한 모델을 동일한 인터페이스로 사용하는 설계 패턴
- 최종 제출: `np.expm1()`으로 로그 역변환 → submission.csv 생성

#### 참조 코드
- `notebook/geo-data-eda-and-feature-engineering.ipynb`: cell-11 (LightGBM 학습)
- `notebook/utils.py`: `get_oof()` (line 473-537), 모델 래퍼 (line 215-470)

---

### 5회차: ML과 DevOps의 접점 - MLOps (2시간)

**목표**: ML 프로젝트의 실무 운영과 DevOps 역할의 접점을 이해한다.

#### 이론 (40분)
- **대회 vs 실무**: 일회성 예측 vs 지속적 학습/배포/모니터링
- **"ML 코드는 전체 시스템의 5%"**: 나머지 95%는 인프라
- **MLOps 파이프라인**:
  - 데이터 파이프라인: ETL, 데이터 품질 관리
  - 학습 파이프라인: 실험 추적, 모델 레지스트리
  - 서빙 파이프라인: 모델 배포, A/B 테스트
  - 모니터링: 데이터 드리프트, 모델 성능 저하 탐지
- **DevOps 엔지니어의 MLOps 역할**:
  - GPU 인프라 관리 (K8s + GPU 노드)
  - ML 파이프라인 CI/CD (Kubeflow, Airflow, MLflow)
  - 모델 서빙 인프라 (TF Serving, Triton)
- **이 프로젝트를 실무에 적용한다면?**:
  - Notebook → Python 스크립트, 수동 → 자동 파이프라인, 로컬 → 분산 학습

#### 실습 & 토론 (50분)
- 이 프로젝트의 ML 파이프라인 다이어그램 함께 그려보기
- 토론: "프로덕션 배포 시 어떤 인프라가 필요한가?"
- 토론: "모델 성능 저하 시 어떤 메트릭을 모니터링해야 하는가?"
- 교육 전체 회고 및 Q&A

#### 참조 자료
- 전 회차 내용 종합

---

## 요약

### 전체 흐름

```
1회차        2회차          3회차          4회차          5회차
ML개론+EDA → 피처엔지니어링 → 지리데이터FE → 모델학습+CV → MLOps
 (기초)       (중급)         (중급)        (중급)       (연결)
```

### DevOps 비유 매핑

| ML 개념 | DevOps 비유 |
|---------|------------|
| 학습 데이터 | 과거 로그/메트릭 |
| 피처 엔지니어링 | 파생 메트릭 생성 |
| 모델 학습 | CI (빌드 + 테스트) |
| 교차 검증 | 스테이징 환경 테스트 |
| 과적합 | 알람이 너무 민감한 상태 |
| 하이퍼파라미터 | 시스템 설정값 |
| 모델 배포 | CD (프로덕션 배포) |
| 데이터 드리프트 | 트래픽 패턴 변화 |

### 프로젝트 파일별 활용

| 파일 | 활용 회차 |
|------|----------|
| `input/train.csv`, `test.csv` | 1, 2회차 |
| `geo-data-eda-and-feature-engineering.ipynb` | 1, 2, 3회차 |
| `Generate Neighbor Info/Stat.ipynb` | 3회차 (개념 설명) |
| `Stacking Ensemble.ipynb` | 4회차 (부분 참조) |
| `utils.py` | 2, 3, 4회차 |

### 사전 준비

- **수강생**: Python 기본 문법, 터미널 사용, `uv` 설치 및 프로젝트 환경 구성
- **교육자**: 노트북 실행 결과 스크린샷, 회차별 요약 슬라이드

### 성공 기준

- EDA 노트북을 독립 실행하고 결과를 해석할 수 있다
- 피처 엔지니어링의 목적과 기법을 설명할 수 있다
- 교차 검증의 원리와 필요성을 이해한다
- ML 프로젝트와 DevOps 업무의 접점을 3가지 이상 말할 수 있다

---

SPEC Version: 2.0.0
Created: 2026-03-02
Updated: 2026-03-02 (v2 - 간소화: 8회차 → 5회차, 앙상블 삭제, 내용 압축)
