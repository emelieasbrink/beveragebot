#Main script for running the project
from furhat_remote_api import FurhatRemoteAPI
from interaction.script import set_gesture,set_persona,bsay,handle_emotional_state,generate_response
from interaction.gestures import create_gestures
from interaction.dictionary import create_keywords_dict, create_responses
from user_perception.video_input import get_pred, video
import cv2
import opencv_jupyter_ui as jcv2
from collections import Counter

FURHAT_IP = "130.243.221.110" # use 127.0.1.1 for Windows


furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)

FACES = {
    'Marty'  : 'Marty'
}

VOICES_EN = {
    'Marty'  : 'GregoryNeural'
}

VOICES_NATIVE = {
    'Marty'  : 'GregoryNeural'
}

def generate_frame_list(list,frame):
    if len(list)==3:
        list.pop(0)
        list.append(frame)
    else:
        list.append(frame)
    return list

def get_pred_from_frame_list(frame_list):
    pred_list = [get_pred(frame)[0] for frame in frame_list]
    counter = Counter(pred_list)
    most_common_pred = counter.most_common(1)[0][0]
    return most_common_pred



def demo_bartender2():
    print('Beveragebot starting...')
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    recording = False

    set_persona('Marty')
    responses_dict = create_responses()
    keywords_dict = create_keywords_dict()
    gestures = create_gestures()
    print('Beveragebot started!')
    bsay("Beveragebot activated, ready to serve")
    frame_list = []
    while(True):
        check, frame = cam.read()
        if (frame is not None):
            frame_list = generate_frame_list(frame_list,frame)
        if not check:
            break
        response = furhat.listen() 
      # costumer_feeling = get_pred_from_frame_list(frame_list) 
        costumer_feeling = get_pred(frame_list[-1])[0]
        print(costumer_feeling)
        generate_response(responses_dict,keywords_dict,response,costumer_feeling, gestures)



if __name__ == '__main__':
    demo_bartender2()        