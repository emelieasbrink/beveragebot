import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from landmarks import read_landmarks_files

data_landmarks= read_landmarks_files()

inputs_landmarks = data_landmarks.drop(columns=['label', 'file'])

print (inputs_landmarks)




