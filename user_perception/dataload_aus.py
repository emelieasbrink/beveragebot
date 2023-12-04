import pandas as pd
from feat import Detector
import cv2
import imageio.v3 as iio
from feat import Detector
from pathlib import Path
import imageio.v3 as iio
import os
import pandas as pd
from feat.utils import FEAT_EMOTION_COLUMNS

def load_data(folder):
    path_name = "./DiffusionFER/DiffusionEmotion_S/" + folder + '/'
    subfolders = [ f.name for f in os.scandir(path_name) if f.is_dir()]

    for emotion in subfolders:
        images = [[], []]
        image_names = []
        path = path_name + emotion + '/'
        for file in Path(path).iterdir():
            if not file.is_file():
                continue
            images[0].append(iio.imread(file))
            images[1].append(emotion)
        image_names = os.listdir(path)
        make_pred(images, 
                  path, 
                  image_names, 
                  folder)
    return

def make_pred(images, 
              path, 
              image_names,
              folder):
    detector = Detector(device="cpu")
    aus_df = pd.DataFrame()
    for i in range(len(image_names)):
        label = images[1][i]
        file = [path + image_names[i]]
        prediction = detector.detect_image(file)
        pred_row = prediction.iloc[0]
        df = pd.DataFrame(pred_row.aus).transpose()
        if(df.isnull().any().any()):
            print('in here')
            continue
        df['file'] = image_names[i].replace(".jpg","",1)
        df['label'] = label
        aus_df = pd.concat([aus_df, df]).reset_index(drop=True)
        
        if (i < 5):
            #test plotting some images for evaluation
            emotion = pred_row[FEAT_EMOTION_COLUMNS].idxmax()
            x = int(pred_row['FaceRectX'])
            y = int(pred_row['FaceRectY'])
            width = int(pred_row['FaceRectWidth'])
            height = int(pred_row['FaceRectHeight'])

            frame = images[0][i]
            cv2.putText(frame, emotion, (x, y - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, 
                        (x, y), 
                        (x + width, y + height), 
                        (0, 255, 0), 2)  
            images_dir = "./processed/" + folder + "/images/" 

            #save the test plots to processes/images
            if not os.path.exists(Path(images_dir)):
                os.makedirs(Path(images_dir))
            iio.imwrite(Path(images_dir + image_names[i]), frame)
        

    file_path_csv = "./processed/" + folder + '/' + label + '_aus.csv'
    aus_df.to_csv(Path(file_path_csv)) 

if __name__ == "__main__":
    load_data('cropped')