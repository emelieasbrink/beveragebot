import cv2
import opencv_jupyter_ui as jcv2
from feat import Detector
from IPython.display import Image
import numpy as np
from joblib import dump, load
import time
import pandas as pd

clf = load('./user_perception/model_train/model.joblib') 
pca = load('./user_perception/model_train/pca.joblib') 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def prepare_features(df, pca):
    """Transforms dataframe columns with pca to match how the model was trained"""
    if pca is not None:
        pca_df = pca.transform(df)
        pca_df = pd.DataFrame(pca_df, columns=['C_1', 
                                            'C_2', 
                                            'C_3', 
                                            'C_4', 
                                            'C_5', 
                                            'C_6',
                                            'C_7'])
        return pca_df
    return df

def get_pred(frame):
    """Based on the frame as input argument, this function classifies the emotion (postive, negative, neutral)
    using pre-trained model and returns it"""
    pred = 'neutral'
    detector = Detector(device="cpu")
    if (frame is not None):
        detected_faces = detector.detect_faces(frame)
        detected_landmarks = detector.detect_landmarks(frame, detected_faces)
        aus = detector.detect_aus(frame, detected_landmarks)
        aus_face = aus[0]
        if (len(aus_face) > 0):
            aus_face_reshape = pd.DataFrame(aus_face[0].reshape(1, -1))
            input = prepare_features(aus_face_reshape, pca)
            pred = clf.predict(input)
    return pred


def video():
    """Test function to see the live classification on yourself.
    Starts the web camera and classifies the emotion based on the pre-trained model"""
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    recording = False

    while True:
        # check = True means we managed to get a frame.
        # If check = False, the device is not available, and we should quit.
        check, frame = cam.read()
        if not check:
            break

        # OpenCV uses a separate window to display output.

        # Press ESC to exit.
        key = jcv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):  # SPACE key
            recording = not recording

        print(get_pred(frame))

        # Iterate through the face coordinates in the current frame
        #for face_coords, emotion in zip(face_coordinates, pred):
        #    x1, y1, x2, y2, confidence = face_coords
        
            # Draw a rectangle around the detected face
            #cv2.putText(frame, emotion, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        jcv2.imshow("video", frame)
        
        #jcv2.imshow("video", frame)

    cam.release()
    jcv2.destroyAllWindows()

if __name__ == "__main__":
    video()