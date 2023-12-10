from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
import numpy as np
import pandas as pd
from feature_sel import read_aus_files
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.discriminant_analysis as skl_da
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

df_full = read_aus_files('full')
df_cropped = read_aus_files('cropped')
print(df_full.shape)

merged = pd.concat([df_full, df_cropped])

file = merged['file'].reset_index(drop=True)
label = merged['label'].reset_index(drop=True)
pca = PCA(n_components=7)
pca_df = pca.fit_transform(merged.drop(columns=['file', 'label']))
print(pca.explained_variance_ratio_.sum())
pca_df = pd.DataFrame(pca_df, columns=['C_1', 
                                       'C_2', 
                                       'C_3', 
                                       'C_4', 
                                       'C_5', 
                                       'C_6',
                                       'C_7'])
pca_df['file'] = file
pca_df['label'] = label

names = pca_df['file'].unique()
np.random.shuffle(names)

image_names_train, image_names_test = train_test_split(names, 
                                                       test_size=0.1)
                                                      # random_state=random_state)

# Create training and testing sets based on the split image names
train_df = pca_df[pca_df['file'].isin(image_names_train)]
test_df = pca_df[pca_df['file'].isin(image_names_test)]

X = train_df.drop(columns=['file', 'label'])
y = train_df['label']
X_test = test_df.drop(columns=['file', 'label'])
y_test = test_df['label']


model_qda = skl_da.QuadraticDiscriminantAnalysis()

#Gridsearch to tune regularisation parameter, reg_param
parameters = {
      'reg_param': (0.000001, 0.00001, 0.0001, 0.001, 0.01), 
      'store_covariance': (True, False),
      'tol': (0.00001, 0.001,0.01, 0.1), 
                    }
#63%
CV_qda = GridSearchCV(
      estimator=model_qda,
      param_grid=parameters,
      n_jobs = -1,
      verbose=2,
      cv = 5
  )
CV_qda.fit(X,y)
qda_best = CV_qda.best_estimator_
qda_pred = qda_best.predict(X_test)
print('QDA')
print(np.mean(qda_pred==y_test))

#cv is used to finetuen and compare models
param_grid = [
    {"kernel":['linear', 'poly', 'rbf', 'sigmoid']},
    {"kernel": ["poly"], "degree":range(1,6)}
]

CV_svm = GridSearchCV(svm.SVC(), 
                          param_grid = param_grid,
                          cv=5,
                          verbose = 2)
print('before')
CV_svm.fit(X, y)
print('after')
# Get the mean test scores (validation accuracy)
cv_results = CV_svm.cv_results_
mean_test_scores = cv_results['mean_test_score']
print('mean test score svm', mean_test_scores)
svm_best = CV_svm.best_estimator_
svm_pred = svm_best.predict(X_test)
print(np.mean(svm_pred==y_test))