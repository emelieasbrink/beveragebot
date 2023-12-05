from time import sleep
from furhat_remote_api import FurhatRemoteAPI
import dictionary as dict
import re
import random
FURHAT_IP = "172.20.10.2"

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

def greeting(emotion):
    if emotion == "happy":
        set_gesture("BigSmile")
        bsay("Hello! You look happy today. You seem to be in the mood for a drink. What kind of drinks you like?")
    elif emotion == "sad":
        set_gesture("ExpressSad")
        bsay("Hello friend! You look like you could need a drink. What can I get you?")
    elif emotion == "angry" or "disgusted":
        set_gesture("Oh")
        set_gesture("BrowRaise")
        set_gesture("Wink")  
        sleep(1)
        bsay("WowWowWow! Maybe you need something to calm down? What kind of drinks do you like?")
    else:
        set_gesture("Wink")        
        bsay("Hello! What kind of drinks do you like?")


def suggest_drink(emotion, answer):
    if "sweet" or "no bitter" in answer.message:
        bsay("Mabye a Mojito?")
    if "sour" in answer.message:
        bsay("Mabye a Lime coctail?")
    if "strong" in answer.message:
        bsay("Mabye a Long island ice tea?")
    if "fresh" or "refreshing" in answer.message:
        bsay("Mabye a water?")


def demo_bartender():
    set_persona('Marty')
    responses_dict = dict.create_responses()
    keywords_dict = dict.create_keywords_dict()

    print("hej")
    while(True):
        response = furhat.listen() 

        print(response)
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
            bsay(responses_dict[key])
        else:
            set_gesture("Oh")
            set_gesture("BrowRaise")
            bsay(random.choice(responses_dict[key]))

if __name__ == '__main__':
    demo_bartender()