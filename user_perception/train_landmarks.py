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
print(data_landmarks)

features = data_landmarks.filter(like='_x')  # select columns ending with '_x'
labels = data_landmarks['label']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.15, random_state=42) #in the parantes original dataset 
X_train, X_validation, y_train, y_validation= train_test_split(X_train, y_train, test_size= 0.2, random_state=42)

# Build and train the kNN model

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
param_grid = {
    'n_neighbors': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

#decission tree classifier
clf= DecisionTreeClassifier().fit(X_train, y_train)
y_pred=clf.predict(X_test)
print (y_pred)
clf.score(X_test, y_test)

#confusion metrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

parameters_cm={
    'max_depth': [10,15], 
    'max_features': ['sqrt', 'log2']}
tree= DecisionTreeClassifier()
clf=GridSearchCV(tree, parameters_cm)
clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)
print(y_pred)

#own crossvalidation
clf=RandomForestClassifier(
    n_estimators=50, 
    max_depth=5, 
    random_state=0)
scores= cross_val_score(clf, X_test, y_test, cv=10 ) #devide into 10 different machine learning models  


grid_search = GridSearchCV(
    estimator=knn_model, 
    param_grid=param_grid, 
    cv=5,
    verbose= 3,
    refit =False)
grid_search.fit(np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test]))

# Make predictions and evaluate the model
predictions_test = knn_model.predict(X_test)
test_score= test_score1 = accuracy_score(y_test, predictions_test) * 100
print("Score for test set 1:", test_score) 






