from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from feature_sel import read_aus_files
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

df = read_aus_files('full')
labels = df['label']
inputs = df.drop(columns=['file', 'label'])

pca = PCA(n_components=7)
pca_df = pca.fit_transform(inputs)
print(pca.explained_variance_ratio_.sum())
pca_df = pd.DataFrame(pca_df, columns=['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7'])

# 90/10
X, X_test, y, y_test = train_test_split(pca_df, labels, test_size=0.1, stratify=labels)

# Gradient Boosting Classifier
model_gb = GradientBoostingClassifier()

# Parameters for GridSearchCV
gb_parameters = {
    'n_estimators': [30, 80, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2, 3, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

CV_gb = GridSearchCV(
    estimator=model_gb,
    param_grid=gb_parameters,
    n_jobs=-1,
    verbose=2,
    cv=5
)

CV_gb.fit(X, y)
gb_best = CV_gb.best_estimator_

# Predictions and evaluation
gb_pred = gb_best.predict(X_test)
print('Gradient Boosting')
print(np.mean(gb_pred == y_test))
