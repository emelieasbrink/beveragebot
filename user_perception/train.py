from model_train.read_aus import read_aus_files, read_aus_files, calculate_valence, read_both_datsets
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV #, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import plt
#from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')


df_full = read_aus_files("./processed/Diffusion/original/")
df_cropped = read_aus_files("./processed/Diffusion/cropped/")
data=df_full

emotion_mapping = {
    'happy': 'positive',
    'surprise': 'positive',
    'angry': 'negative',
    'fear': 'negative',
    'sad': 'negative',
    'disgust': 'negative',
    'neutral': 'neutral'
}
data['emotion_category'] = data['label'].map(emotion_mapping)
data['label'] = data['emotion_category']
data.drop(columns=['emotion_category'], inplace=True, errors='ignore')
file = data['file'].reset_index(drop=True)
label = data['label'].reset_index(drop=True)

inputs = data.drop(columns=['file','label'])  # Exclude 'file' column
scaler = StandardScaler()
features_scaled = scaler.fit_transform(inputs)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.15, random_state=42) #in the parantes original dataset 
X_train, X_validation, y_train, y_validation= train_test_split(X_train, y_train, test_size= 0.2, random_state=42)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
param_grid = {
    'n_neighbors': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}

# Cross-validated Accuracy 0.71
#knn grid search 
grid_search_knn = GridSearchCV(
    estimator=knn_model, 
    param_grid=param_grid, 
    cv=5,
    verbose= 3,
    refit =True)
grid_search_knn.fit(np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]))

#evaluate
scores = cross_val_score(grid_search_knn, np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]), cv=5)
print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Decision treeclassifier: 57.73195876288659
#decision tree  
param_grid_dts={
    'max_depth': [10,15], 
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    }

tree= DecisionTreeClassifier()
clf_dts=GridSearchCV(
    estimator=tree,
    param_grid= param_grid_dts,
    )
clf_dts.fit(X_train, y_train)

#evaluate
y_pred= clf_dts.predict(X_test)
test_score_dts = accuracy_score(y_test, y_pred) * 100
clf_dts_score=clf_dts.score(X_test, y_test) #compare pred value with the actual value 
c_report= classification_report(y_test, y_pred)
print(f"Decision treeclassifier: {test_score_dts}")
print(c_report)

param_grid_rfc= {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
}

#random forest calssifier
#Cross-Validation accuracy scoreScores: 73.19587628865979 
# 0.73 (+/- 0.06) 
rfc=RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=0)

grid_search_rfc = GridSearchCV(
    rfc, 
    param_grid_rfc)
grid_search_rfc.fit(X_train, y_train)
y_pred_rfc = grid_search_rfc.predict(X_test)

#evaluate
test_score_rfc = accuracy_score(y_test, y_pred_rfc) * 100
clf_rf_score = grid_search_rfc.score(X_test, y_test)
c_report_rf = classification_report(y_test, y_pred_rfc)

scores_rfc = cross_val_score(grid_search_rfc, np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]), cv=5)
print("Cross-validated Accuracy rfc: %0.2f (+/- %0.2f)" % (scores_rfc.mean(), scores_rfc.std() * 2))
print("Random Forest Classifier Cross-Validation accuracy scoreScores:", test_score_rfc)
print(c_report_rf)


#linear regression
#Mean Squared Error: 5.651821056679385e-31
data1 = df_full
print (data1)
emotion_mapping1 = {
    'positive': 1,
    'negative': -1,
    'neutral': 0
}

data1['emotion_category'] = data1['label'].map(emotion_mapping1)
print ((data1))
data1['label'] = data1['emotion_category'] #replaces

inputs1 = data1.drop(columns=['file','label']) 
target1 = data1['emotion_category']


X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs1, target1, test_size=0.15, random_state=42)
X_train1, X_validation1, y_train1, y_validation1 = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train1, y_train1)
y_pred1 = linear_model.predict(X_test1)
mse = mean_squared_error(y_test1, y_pred1)
#print(f"Linear regresssion pred: {y_pred1}")
print(f"Mean Squared Error: {mse}")


      
