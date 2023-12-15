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


def df_to_csv(df,
              path):
    
    """
    Writes a dataframe to a csv file. 
    If file exists it adds the dataframe to the existing file. 
    If it does not exist it creates a new file and adds the dataframe.
    
    Arguments: 
    df - dataframe to add to csv file
    path - path for csv file to be stored
    """

    if os.path.exists(path):
        existing_data = pd.read_csv(path)
        combined_data = pd.concat([existing_data, df], ignore_index=True)
        combined_data.to_csv(path, 
                             index=False)
    else:
        df.to_csv(path, 
                  index=False)
    
def find_label_degree(degree):
    """Converts labels in MultiEmoVA into only positive, negative and neutral"""

    if 'positive' in degree.lower():
        return('positive')
    elif 'neutral' in degree.lower():
        return('negative')
    else:
        return('neutral')
    
def load_both_datasets(folder):
    """
    Calls functions that load data.
    folder can take three arguments ['all', 'cropped', 'original']
    If the argument is all it reads but cropped and orginal images from DiffusionFER, 
    Otherwise it only reads files from the stated folder.
    """

    if folder == 'all':
        load_data("./DiffusionFER/DiffusionEmotion_S/original/", 
              'original', 
              True)
        load_data("./DiffusionFER/DiffusionEmotion_S/cropped/", 
              'cropped', 
              True)
        load_data("./MultiEmoVA/", Diff=False)
    else:
        load_data("./DiffusionFER/DiffusionEmotion_S/" + folder + '/', 
                folder, 
                True)
        load_data("./MultiEmoVA/", Diff=False)
    return

def load_data(path_name,
              folder = 'cropped',
              Diff = True):
    """
    Loads images from path, calls make_pred to detect faces and store action units. 
    Takes the following arguments:
    path_name - ['./DiffusionFER/DiffusionEmotion_S/original/', 
                "./DiffusionFER/DiffusionEmotion_S/cropped/",
                "./MultiEmoVA/"] depending on which images to load
    folder - ['cropped', 'full'], default is cropped. Only relevant for the diffusionFER dataset and states the name of the folder that contains the images.
    Diff - [True, False], default it True. States if the data is loaded from the diffusionFER dataset or not. 
    """

    subfolders = [ f.name for f in os.scandir(path_name) if f.is_dir()]

    for emotion in subfolders:
        images = [[], []]
        image_names = []

        if not Diff:
            label = find_label_degree(emotion)
        else:
            label = emotion

        path = path_name + emotion + '/'
        for file in Path(path).iterdir():
            if not file.is_file():
                continue
            images[0].append(iio.imread(file))
            images[1].append(label)
        image_names = os.listdir(path)
        make_pred(images, 
                  path, 
                  image_names, 
                  folder,
                  Diff)
    return

def make_pred(images, 
              path, 
              image_names,
              folder,
              Diff):
    """
    Called from load_dataset. 
    Detects faces in images and store them is aus files in the directory processed. 
    Also saves 5 images from each label to be able to see if the face detecting works as intended. 
    """
    detector = Detector(device="cpu")
    aus_df = pd.DataFrame()
    if Diff:
        dataset= 'Diffusion/' + folder
    else:
        dataset='Multi/'

    dir = "./processed/" + dataset 
    if not os.path.exists(Path(dir)):
        os.makedirs(Path(dir))
    
    for i in range(len(image_names)):
        label = images[1][i]
        file = [path + image_names[i]]
        prediction = detector.detect_image(file)
        
        frame = None
        if (prediction.shape[0] < 10):
            for index, row in prediction.iterrows():
                df = pd.DataFrame(row.aus).transpose()
                if(df.isnull().any().any()):
                    print('no face detected')
                    continue
                df['file'] = image_names[i].replace(".jpg","",1)
                df['face'] = index
                df['label'] = label
                aus_df = pd.concat([aus_df, df]).reset_index(drop=True)
            
                if (i < 5):
                    #test plotting some images for evaluation
                    emotion = row[FEAT_EMOTION_COLUMNS].idxmax()
                    x = int(row['FaceRectX'])
                    y = int(row['FaceRectY'])
                    width = int(row['FaceRectWidth'])
                    height = int(row['FaceRectHeight'])

                    frame = images[0][i]
                    cv2.putText(frame, emotion, (x, y - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, 
                                (x, y), 
                                (x + width, y + height), 
                                (0, 255, 0), 2)  

            images_dir = "./processed/" + dataset + "/images/" 

            #save the test plots to processes/images
            if not os.path.exists(Path(images_dir)):
                os.makedirs(Path(images_dir))
            if (i < 5 and frame is not None):
                iio.imwrite(Path(images_dir + image_names[i]), frame)
        

    file_path_csv = "./processed/" + dataset + label + '_aus.csv'
    df_to_csv(aus_df, 
              Path(file_path_csv))

if __name__ == "__main__":
    load_data("./MultiEmoVA/", Diff=False)