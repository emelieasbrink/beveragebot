import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from feature_sel import read_aus_files

data = read_aus_files()
print(data.shape)

labels = data['label']
inputs = data.drop(columns=['label', 'file'])  # Exclude 'file' column
#inputs= data ['AU01', 'AU02', 'AU03', 'AU04', 'AU05']
#inputs = data.iloc[:, :5] 

# Split data into training, validation, and test sets
train_in, test_in, train_out, test_out = train_test_split(
    inputs, labels, test_size=0.1, random_state=42, stratify=labels
)

train_features, val_features, train_labels, val_labels = train_test_split(
    train_in, train_out, test_size=(0.2/0.9), random_state=42, stratify=train_out
)

# Create and train the KNeighborsClassifier model
knn_model = KNeighborsClassifier()
knn_model.fit(train_features, train_labels)

# Use PredefinedSplit for the combined train and validation sets
split_index = [-1 if x in train_features.index else 0 for x in pd.concat([train_features, val_features]).index]
ps = PredefinedSplit(split_index)

param_grid = {
    'n_neighbors': [65, 70, 72, 77, 78, 80],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    cv=ps,
    verbose=3,
    refit=False
)
grid_search.fit(pd.concat([train_features, val_features]), pd.concat([train_labels, val_labels]))

# Use the best model obtained from GridSearchCV
best_knn_model = KNeighborsClassifier(**grid_search.best_params_)
best_knn_model.fit(pd.concat([train_features, val_features]), pd.concat([train_labels, val_labels]))

# Evaluate the best model on the test set
test_predictions = best_knn_model.predict(test_in)
test_score = accuracy_score(test_out, test_predictions) * 100
print("Score for test set:", test_score)


   # Evaluate the best model on the test set
test_predictions = best_knn_model.predict(test_in)
val_predictions= best_knn_model.predict(val_features)
train_predictions= best_knn_model.predict(train_in)
test_score1 = accuracy_score(test_out, test_predictions) * 100
val_score1=accuracy_score(val_labels, val_predictions) * 100
train_scare1=accuracy_score(train_out, train_predictions) * 100
print("Score for train set 1:", train_scare1)  
print("Score for validation set1:", val_score1)  
print("Score for test set:1", test_score1) 
print(
        "This is with the 'best' params of:", grid_search.best_params_
    )



