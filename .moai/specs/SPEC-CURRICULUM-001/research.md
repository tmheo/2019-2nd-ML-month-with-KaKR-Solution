# Research: ML Education Curriculum using House Price Prediction Project

## Project Analysis

### Project Overview
- **Competition**: 2019 2nd ML month with KaKR (Kaggle Korea)
- **Task**: House price prediction (regression)
- **Result**: 11th place (Public: 98,316 / Private: 99,336 RMSE)
- **Dataset**: Ames Housing Dataset (~21,600 train / ~6,900 test / 79 features)

### Notebook Structure (Execution Order)
1. `geo-data-eda-and-feature-engineering.ipynb` - EDA & geo feature engineering
2. `Generate Neighbor Info.ipynb` - Haversine distance neighbor calculation
3. `Generate Neighbor Stat.ipynb` - Neighbor statistics aggregation
4. `Stacking Ensemble.ipynb` - Multi-stage stacking ensemble

### Key ML Concepts Covered in This Project

#### 1. Data Exploration (EDA)
- Train/test data loading and basic statistics
- Distribution visualization (histplot, scatterplot, boxplot)
- Outlier detection and removal
- Feature importance analysis

#### 2. Feature Engineering
- **Temporal features**: yr_sold, qt_sold, week_sold, dow_sold
- **Ratio features**: bathrooms/bedrooms, sqft_living/bedrooms
- **Boolean features**: has_basement, is_renovated
- **Interaction features**: sqft_living * grade, sqft_living * overall
- **Zipcode decomposition**: 5-digit split into multiple sub-features
- **Target encoding**: Group-by price statistics per category

#### 3. Geo-Spatial Feature Engineering
- PCA transformation on lat/long coordinates
- K-Means clustering on coordinates (Elbow method + CV-based K selection)
- Bearing angle calculation from center
- Haversine distance (great-circle distance on sphere)
- Neighbor-based statistics (1km/3km/5km radius, 5/10/20 nearest)

#### 4. Data Preprocessing
- Log transformation (np.log1p) for skewed target
- Box-Cox transformation for skewed features
- RobustScaler for feature normalization
- One-Hot Encoding for categorical variables
- Label Encoding alternative path
- Memory optimization (reduce_mem_usage)

#### 5. Model Training & Validation
- K-Fold Cross Validation (5-fold)
- Out-of-Fold (OOF) prediction pattern
- Early stopping for boosting models
- RMSE evaluation metric

#### 6. Models Used
- **Linear**: Ridge, Lasso, ElasticNet
- **Tree-based**: RandomForest, ExtraTrees, GradientBoosting
- **Boosting**: XGBoost, LightGBM, CatBoost
- **Neural Network**: Keras Dense NN, Embedding NN
- **SVM**: SVR

#### 7. Ensemble Methods
- **Stage 1**: 9 models x 8 feature sets = 72 OOF predictions
- **Stage 2**: 18 meta-models on Stage 1 outputs
- **Final**: Linear blending (ElasticNet) on Stage 2 outputs

### Code Architecture (utils.py)
- Model wrapper pattern (SklearnWrapper, XgbWrapper, LgbmWrapper, CatWrapper, KerasWrapper)
- Unified train/predict interface across all model types
- `get_oof()` function for OOF prediction pipeline
- `stacking()` function for multi-stage ensemble
- `load_data()` function with configurable feature sets
- Utility functions: rmse, haversine_array, bearing_array, groupby_helper

### Target Audience Analysis
- DevOps engineers (junior to 5-6 years experience)
- No ML background assumed
- Strong in: Linux, networking, CI/CD, containers, scripting
- Familiar with: Python (basic), data formats (JSON/YAML/CSV)
- Need: Conceptual understanding + hands-on practice
