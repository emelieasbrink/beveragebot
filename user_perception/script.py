from datasets import load_dataset
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from feat import Detector
from statistics import mean
import matplotlib.pyplot as plt

from feat.utils import FEAT_EMOTION_COLUMNS


dataset = load_dataset("FER-Universe/DiffusionFER")


#print(dataset['train'])

training_data = dataset['train']
labels = training_data['label']
label_counts = Counter(labels)

first_five_images = pd.DataFrame
#image_name = os.path.basename(training_data[0])

print("Label\tCount")
print("-----------------")
for label, count in label_counts.items():
    print(f"{label}\t{count}")
   

first_image_path = training_data['image'][0]
first_image_name = os.path.basename(first_image_path)

print(f"First Image Name: {first_image_name}")