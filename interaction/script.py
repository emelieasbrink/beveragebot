from time import sleep
from furhat_remote_api import FurhatRemoteAPI
import dictionary as dict
from gestures import create_gestures
import re
import random

FURHAT_IP = "130.243.218.200" # use 127.0.1.1 for Windows


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



def set_persona(persona):
    furhat.set_face(character=FACES[persona], mask="Adult")
    furhat.set_voice(name=VOICES_EN[persona])
    furhat.set_led(red=50, green=50, blue=100)

def bsay(line):
    furhat.say(text=line, blocking=True)

def set_gesture(emotion):
    furhat.gesture(name=emotion)


    
def handle_emotional_state(): ##Detta ska kombineras me 
    return "positive"
    

def generate_response(responses_dict,keywords_dict,response,customer_feeling, gestures):
        if response.success==False:
            print('Something went wrong while listenting!')
        else:    
            print(f"Beveragebot perceived: '{response.message}'")
        matched_intent = None

        for intent,pattern in keywords_dict.items():
        # Using the regular expression search function to look for keywords in user input
            if re.search(pattern, response.message): 
            # if a keyword matches, select the corresponding  
                matched_intent=intent  
        # The fallback intent is selected by default
        key='fallback' 
        if matched_intent in responses_dict:
            key = matched_intent
            if customer_feeling == 'positive' and 'positive_responses' in responses_dict[key]:
                gesture = random.choice(gestures['positive'])
                print(f"Furhat gesture: {gesture['name']}")
                furhat.gesture(body=gesture)
                bsay(random.choice(responses_dict[key]['positive_responses']))
            elif customer_feeling == 'negative' and 'negative_responses' in responses_dict[key]:
                gesture = random.choice(gestures['negative'])
                print(f"Furhat gesture: {gesture['name']}")
                furhat.gesture(body=gesture)
                bsay(random.choice(responses_dict[key]['negative_responses']))
            elif customer_feeling == 'neutral' and 'neutral_responses' in responses_dict[key]:
                gesture = random.choice(gestures['neutral'])
                print(f"Furhat gesture: {gesture['name']}")
                furhat.gesture(body=gesture)
                bsay(random.choice(responses_dict[key]['neutral_responses']))
        else:
            set_gesture("Thoughtful")
            bsay(random.choice(responses_dict[key].get('fallback_responses', []))) 


def demo_bartender():
    print('Beveragebot starting...')
    set_persona('Marty')
    responses_dict = dict.create_responses()
    keywords_dict = dict.create_keywords_dict()
    gestures = create_gestures()
    print('Beveragebot started!')
    bsay("Beveragebot activated, ready to serve")
    while(True):
        response = furhat.listen() 
        costumer_feeling = handle_emotional_state() 
        generate_response(responses_dict,keywords_dict,response,costumer_feeling, gestures)



if __name__ == '__main__':
    demo_bartender()