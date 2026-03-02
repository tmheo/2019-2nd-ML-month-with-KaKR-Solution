# Acceptance Criteria: SPEC-UPDATE-001

## Scenario 1: uv 환경 설정 및 의존성 설치

```gherkin
Given pyproject.toml이 프로젝트 루트에 존재하고
  And .python-version 파일이 존재할 때
When `uv sync`를 실행하면
Then .venv 디렉토리가 생성되고
  And 모든 의존성이 성공적으로 설치되어야 한다
```

## Scenario 2: Jupyter 커널 등록 및 노트북 실행

```gherkin
Given uv 가상 환경이 설정되고
  And ipykernel이 설치되어 있을 때
When Jupyter 커널 목록을 확인하면
Then "Kaggle 2019 (uv)" 커널이 목록에 표시되어야 한다
```

## Scenario 3: utils.py import 성공

```gherkin
Given utils.py의 deprecated API가 모두 현대화되었을 때
When uv 환경에서 `from utils import *`를 실행하면
Then ImportError 없이 모든 모듈이 로드되어야 한다
```

## Scenario 4: 노트북 커널 호환성

```gherkin
Given 모든 노트북의 deprecated API가 현대화되었을 때
When 각 노트북의 첫 번째 코드 셀을 실행하면
Then DeprecationWarning이나 ImportError가 발생하지 않아야 한다
```

## Edge Cases

- Python 3.14에서 TensorFlow 설치 실패 시 → .python-version을 3.12로 설정
- catboost가 최신 numpy와 호환 불가 시 → numpy 버전 상한 설정
- seaborn histplot이 distplot과 다른 파라미터를 요구할 시 → 파라미터 매핑 적용

## Quality Gates

- uv sync 성공 (exit code 0)
- utils.py import 성공 (no exceptions)
- Jupyter 커널 등록 확인
- 4개 노트북 모두 첫 셀 실행 성공
