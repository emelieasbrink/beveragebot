#Main script for running the project
from interaction import script as scr
from furhat_remote_api import FurhatRemoteAPI
from interaction import dictionary as dict
from interaction import create_gestures

FURHAT_IP = "127.0.1.1" # use 127.0.1.1 for Windows


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



def demo_bartender():
    print('Beveragebot starting...')
    scr.set_persona('Marty')
    responses_dict = scr.dict.create_responses()
    keywords_dict = scr.dict.create_keywords_dict()
    gestures = scr.create_gestures()
    print('Beveragebot started!')
    scr.bsay("Beveragebot activated, ready to serve")
    while(True):
        response = furhat.listen() 
        costumer_feeling = scr.handle_emotional_state() 
        scr.generate_response(responses_dict,keywords_dict,response,costumer_feeling, gestures)

if __name__ == '__main__':
    demo_bartender()        