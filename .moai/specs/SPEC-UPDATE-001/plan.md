# Implementation Plan: SPEC-UPDATE-001

## Task Decomposition

### Task 1: pyproject.toml 생성
- File: `pyproject.toml` (new)
- Action: Create with project metadata, dependencies, and tool configurations
- Dependencies: None

### Task 2: .python-version 생성
- File: `.python-version` (new)
- Action: Set Python version for uv (3.12 recommended for ML library compatibility)
- Dependencies: None

### Task 3: .gitignore 업데이트
- File: `.gitignore` (existing)
- Action: Add `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
- Dependencies: None

### Task 4: utils.py 현대화
- File: `notebook/utils.py` (existing)
- Action: Fix all deprecated imports and API calls
- Changes:
  1. sklearn.preprocessing.Imputer → sklearn.impute.SimpleImputer
  2. keras.* → tensorflow.keras.*
  3. tensorflow.set_random_seed → tf.random.set_seed
  4. tf.ConfigProto/Session → Remove (TF2 eager mode)
  5. tqdm.tqdm_notebook → tqdm.notebook.tqdm
- Dependencies: None

### Task 5: geo-data-eda notebook 현대화
- File: `notebook/geo-data-eda-and-feature-engineering.ipynb` (existing)
- Action: Fix sns.distplot → sns.histplot calls
- Dependencies: None

### Task 6: Stacking Ensemble notebook 현대화
- File: `notebook/Stacking Ensemble.ipynb` (existing)
- Action: Remove normalize parameter from Ridge/Lasso/ElasticNet
- Dependencies: Task 4 (utils.py must be fixed first)

### Task 7: uv sync 및 Jupyter 커널 등록
- Action: Run `uv sync`, register ipykernel
- Dependencies: Task 1, Task 2

### Task 8: 실행 검증
- Action: Import test for utils.py, notebook kernel availability check
- Dependencies: Task 4, Task 7

## Execution Order

```
[Task 1, 2, 3] (parallel - no dependencies)
     ↓
[Task 4, 5] (parallel - independent file modifications)
     ↓
[Task 6] (depends on Task 4)
     ↓
[Task 7] (depends on Task 1, 2)
     ↓
[Task 8] (verification)
```

## Technology Stack

- Package Manager: uv 0.10.x
- Python: 3.12.x (via uv, for ML library compatibility)
- Virtual Environment: uv managed (.venv/)
- Jupyter: ipykernel registered in uv environment
