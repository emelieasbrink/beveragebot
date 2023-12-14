import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV #, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from landmarks import read_landmarks_files
from feat import Detector
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
#from landmarks import read_landmarks_files

#from matplotlib import pyplot as plt

detector = Detector(device= "auto")
data_landmarks= read_landmarks_files()
#print (data_landmarks)

# Extract landmark numbers from existing columns
landmark_numbers = [int(col.split('_')[1]) for col in data_landmarks.columns if 'landmark' in col]

# Create new columns for x and y coordinates
for num in landmark_numbers:
    x_col = f'landmark_{num}_x'
    y_col = f'landmark_{num}_y'
    
    # Extract x and y coordinates from the existing columns
    data_landmarks[x_col] = data_landmarks[f'landmark_{num}'].apply(lambda x: eval(x)[0])
    data_landmarks[y_col] = data_landmarks[f'landmark_{num}'].apply(lambda x: eval(x)[1])

# Drop the original landmark columns
data_landmarks = data_landmarks.drop(columns=[f'landmark_{num}' for num in landmark_numbers])
data_landmarks = data_landmarks.drop(columns="file")

features = data_landmarks.filter(like='_x')  # select columns ending with '_x'
labels = data_landmarks['label']

emotion_mapping = {
    'happy': 'positive',
    'surprise': 'positive',
    'angry': 'negative',
    'fear': 'negative',
    'sad': 'negative',
    'disgust': 'negative',
    'neutral': 'neutral'
}

data_landmarks['emotion_category'] = data_landmarks['label'].map(emotion_mapping)
print(data_landmarks)
data_landmarks['label'] = data_landmarks['emotion_category']
data_landmarks.drop(columns=['emotion_category'], inplace=True, errors='ignore')
label = data_landmarks['label'].reset_index(drop=True)

#scaler = StandardScaler()
#features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
inputs = data_landmarks.drop(columns=['label'])  # Exclude 'file' column
scaler = StandardScaler()
features_scaled = scaler.fit_transform(inputs)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.15, random_state=42) #in the parantes original dataset 
X_train, X_validation, y_train, y_validation= train_test_split(X_train, y_train, test_size= 0.2, random_state=42)

#kNN model
#Cross-validated Accuracy: 0.55 (+/- 0.09)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
param_grid = {
    'n_neighbors': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['ball_tree', 'kd_tree', 'brute']
}

grid_search_knn = GridSearchCV(
    estimator=knn_model, 
    param_grid=param_grid, 
    cv=5,
    verbose= 3,
    refit =True)
grid_search_knn.fit(np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]))

scores = cross_val_score(grid_search_knn, np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]), cv=5)
print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#random forest calssifier
#Cross-validated Accuracy rfc: 0.61 (+/- 0.05)
#Random Forest Classifier Cross-Validation accuracy scoreScores: 63.33333333333333
param_grid_rfc= {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
}

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

 






