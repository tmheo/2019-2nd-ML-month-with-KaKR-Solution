# Research: 2019 Kaggle Project Modernization

## Project Overview

2019 Kaggle Korea House Price Prediction competition (11th place solution).
Repository contains 4 Jupyter notebooks and 1 shared Python utility file.

## File Inventory

### Notebooks (notebook/)
1. `geo-data-eda-and-feature-engineering.ipynb` (4.8MB) - Standalone EDA notebook
2. `Generate Neighbor Info.ipynb` (44KB) - Haversine distance neighbor calculation
3. `Generate Neighbor Stat.ipynb` (163KB) - Neighbor statistics aggregation
4. `Stacking Ensemble.ipynb` (312KB) - Multi-level stacking ensemble model

### Python Files
- `notebook/utils.py` (36KB) - Shared ML utilities, model wrappers, feature engineering

### Data Files (input/)
- `train.csv`, `test.csv` - Original competition data
- `neighbor_*_stat.csv` - Pre-computed neighbor statistics (6 files)
- `nearest_*_neighbor_stat.csv` - Pre-computed nearest neighbor stats (3 files)
- `x_train_*.csv`, `x_test_*.csv` - Intermediate stacking features
- `cv_score_*.csv` - Cross-validation scores

## Dependency Analysis

### Core Dependencies (from utils.py and notebooks)

| Package | Import Pattern | Current API | Issue |
|---------|---------------|-------------|-------|
| pandas | `import pandas as pd` | Standard | Compatible |
| numpy | `import numpy as np` | Standard | Compatible |
| matplotlib | `import matplotlib.pyplot as plt` | Standard | Compatible |
| seaborn | `sns.distplot()` | Deprecated in 0.12+ | Must use `histplot`/`kdeplot` |
| scikit-learn | `from sklearn.preprocessing import Imputer` | Removed in 1.0+ | Must use `SimpleImputer` |
| scikit-learn | `Ridge(normalize=True)` | Removed in 1.2+ | Must use Pipeline + StandardScaler |
| xgboost | `import xgboost as xgb` | Standard | Compatible |
| lightgbm | `import lightgbm as lgb` | Standard | Compatible |
| catboost | `import catboost as cat` | Standard | Compatible |
| tensorflow | `tf.ConfigProto`, `tf.Session`, `set_random_seed` | TF1 API removed in TF2 | Major rewrite needed |
| keras | `from keras.models import Sequential` | Standalone Keras | Must use `tf.keras` or `keras>=3.0` |
| scipy | `from scipy.stats import norm, skew` | Standard | Compatible |
| psutil | `import psutil` | Standard | Compatible |
| tqdm | `from tqdm import tqdm_notebook as tqdm` | Renamed | Must use `tqdm.notebook.tqdm` |

### Deprecated/Removed API Details

#### 1. TensorFlow 1.x API (utils.py:52-73)
```python
# CURRENT (broken)
from tensorflow import set_random_seed
session_conf = tf.ConfigProto(...)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# REQUIRED (TF2)
tf.random.set_seed(RANDOM_SEED)
# Session/ConfigProto not needed - TF2 uses eager execution
```

#### 2. Standalone Keras Imports (utils.py:43-49)
```python
# CURRENT (broken)
from keras.models import Sequential, Model
from keras.layers import Dense, ...
from keras import optimizers

# REQUIRED
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ...
from tensorflow.keras import optimizers
```

#### 3. sklearn Imputer (utils.py:30)
```python
# CURRENT (broken)
from sklearn.preprocessing import Imputer

# REQUIRED
from sklearn.impute import SimpleImputer
```

#### 4. sklearn normalize parameter (Stacking Ensemble.ipynb)
```python
# CURRENT (broken)
ridge_param = {'normalize': True, ...}
lasso_param = {'normalize': True, ...}
elastic_param = {'normalize': True, ...}

# REQUIRED: Remove normalize param, use Pipeline if needed
```

#### 5. seaborn distplot (geo-data-eda notebook, utils.py)
```python
# CURRENT (deprecated)
sns.distplot(data, ...)

# REQUIRED
sns.histplot(data, kde=True, ...) or sns.kdeplot(data, ...)
```

#### 6. tqdm_notebook (utils.py:60)
```python
# CURRENT (deprecated)
from tqdm import tqdm_notebook as tqdm

# REQUIRED
from tqdm.notebook import tqdm
```

#### 7. keras.utils.plot_model (utils.py:49)
```python
# CURRENT (broken)
from keras.utils import plot_model

# REQUIRED
from tensorflow.keras.utils import plot_model
```

## Python 3.14 Compatibility Notes

- Python 3.14 is bleeding edge. Some ML libraries may not have wheels yet.
- Recommended: Use Python 3.12 or 3.13 for maximum compatibility.
- Key concern: TensorFlow may not support Python 3.14 yet.
- XGBoost, LightGBM, CatBoost generally lag behind Python releases.

## Notebook Execution Order

The notebooks have an implicit execution dependency:
1. `geo-data-eda-and-feature-engineering.ipynb` (standalone, EDA only)
2. `Generate Neighbor Info.ipynb` → produces `neighbor_info.csv`
3. `Generate Neighbor Stat.ipynb` → produces `neighbor_*_stat.csv` files
4. `Stacking Ensemble.ipynb` → final model (uses all generated CSV files)

Note: Steps 2-3 produce intermediate CSV files already present in `input/`.
The stacking notebook can run using pre-computed data.

## Architecture Pattern

- `utils.py` provides model wrapper classes (XgbWrapper, LgbmWrapper, CatWrapper, KerasWrapper, KerasEmbeddingWrapper, SklearnWrapper)
- OOF (Out-of-Fold) prediction framework with `get_oof()` function
- Multi-level stacking ensemble pipeline via `stacking()` function
- Feature engineering with geographic data (Haversine distance, PCA, KMeans clustering)
