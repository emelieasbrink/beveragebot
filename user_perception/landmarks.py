import cv2
import dlib
import pandas as pd
from pathlib import Path
import os
import imageio.v3 as iio


def read_landmarks_files():
    path_landmark = "./processed/"
    all_landmark = pd.DataFrame()

    for file in Path(path_landmark).iterdir():
        if not file.is_file():
            continue
        if 'csv' in str(file):
            landmark_df = pd.read_csv(file)
            all_landmark = pd.concat([landmark_df, all_landmark]).reset_index(drop=True)
    all_landmark = all_landmark.drop(columns='Unnamed: 0')
    return all_landmark

def load_data():
    path_name = "./DiffusionFER/DiffusionEmotion_S/cropped/"
    subfolders = [f.name for f in os.scandir(path_name) if f.is_dir()]

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
        make_pred_landmarks(images, path, image_names)
    return

def make_pred_landmarks(images, path, image_names):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")  # Replace with the path to your shape predictor file

    landmark_df = pd.DataFrame()
    for i in range(len(image_names)):
        label = images[1][i]
        file = path + image_names[i]
        frame = images[0][i]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 1:
            landmarks = predictor(gray, faces[0])
            landmarks_array = [(p.x, p.y) for p in landmarks.parts()]
            landmarks_df = pd.DataFrame([landmarks_array], columns=[f"landmark_{i}" for i in range(1, 69)])
            landmarks_df['file'] = image_names[i].replace(".jpg", "", 1)
            landmarks_df['label'] = label
            landmark_df = pd.concat([landmark_df, landmarks_df]).reset_index(drop=True)

            if i < 5:
                # Test plotting for evaluation (you can adjust this part)
                for (x, y) in landmarks_array:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                images_dir = "./processed/landmarks/"
                if not os.path.exists(Path(images_dir)):
                    os.makedirs(Path(images_dir))
                iio.imwrite(Path(images_dir + image_names[i]), frame)

    file_path_csv = "./processed/landmarksCsv/" + label + '_landmark.csv'
    landmark_df.to_csv(Path(file_path_csv), index=False)
    
    print(landmark_df)
    return landmark_df

if __name__ == "__main__":
    load_data()
