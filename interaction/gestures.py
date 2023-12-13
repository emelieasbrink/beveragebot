from numpy.random import randint
from numpy.random import uniform

def create_gestures():
# Building a dictionary of gestures
    gestures = {

# Neutral gestures #2
        'natural': [
        {
                "name":"NeutralFace",
                "frames" : 
                [{
                    "time" : [0.32, 2],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : randint(-4,4),
                        "NECK_TILT" : randint(-4,4),
                        "NECK_ROLL" : randint(-4,4),
                        
                         
                
                    }
                },
                {
                    "time":[2.32],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],

            "class": "furhatos.gestures.Gesture"
            },

            {
                "name":"SimpleNodding",
                "frames" : 
                [{
                    "time" : [0.32, 1],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : 0,
                        "NECK_TILT": 5,
                        "NECK_ROLL" : 0,
                        "LOOK_LEFT" : 0,
                        "LOOK_RIGHT": 0,
                         
                        
                    }
                },
                {
                    "time" : [1, 1.62],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : 0,
                        "NECK_TILT": 0,
                        "NECK_ROLL" :0,
                        "LOOK_LEFT" : 0,
                        "LOOK_RIGHT": 0,
                         
                        
                    }
                },
                {
                    "time" : [1.62, 2.22],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : 0,
                        "NECK_TILT": 5,
                        "NECK_ROLL" : 0,
                       
                    }
                },
                {
                    "time":[2.62],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],

            "class": "furhatos.gestures.Gesture"
            },
        ],

# Positive Gestures #2
        'positive': [

            {   "name":"OpenSmile",
                "frames" : 
                    [{
                    "time" : [0.33, 2],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : randint(-4,4),
                        "NECK_TILT" : randint(-4,4),
                        "NECK_ROLL" : randint(-4,4),
                        "SMILE_OPEN": 0.8,
                    }
                },
                {
                    "time":[2.33],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }
                
                ],

            "class": "furhatos.gestures.Gesture"
            },
            {
                "name":"BigSmile",
                "frames":[
                    {
                    "time":[0.32,2],
                    "persist":False, 
                    "params":{
                        "BROW_UP_LEFT":1,
                        "BROW_UP_RIGHT":1,
                        "SMILE_OPEN":0.1,
                        "SMILE_CLOSED":0.7
                        }
                    },
                    {
                    "time":[2.32],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],
                "class":"furhatos.gestures.Gesture"
                },
        
        ],

# Negative Gestures #3
        'negative': [
            {
                "name": "JustSad",
                "frames" : 
                [{
                    "time" : [0.32, 3],
                    "persist" : False,
                    "params": {
                        "NECK_PAN"  : randint(-4,4),
                        "NECK_TILT" : randint(-4,4),
                        "NECK_ROLL" : randint(-4,4),
                        "EXPR_SAD": 0.8,
                    }
                },
                {
                    "time":[3.32],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],

            "class": "furhatos.gestures.Gesture"
            },
            {
                "name":"SadSmile",
                "frames":[
                    {
                    "time":[0.32,3],
                    "persist":False, 
                    "params":{
                        "NECK_PAN"  : randint(-4,4),
                        "NECK_TILT" : randint(-4,4),
                        "NECK_ROLL" : randint(-4,4),
                        "EXPR_SAD": 0.6,
                        "SMILE_CLOSED":0.6
                        
                        }
                    },
                    {
                    "time":[3.32],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],
                "class":"furhatos.gestures.Gesture"
                },
                {
                "name":"Worried",
                "frames":[
                    {
                    "time":[0.32,3],
                    "persist":False, 
                    "params":{
                        "NECK_PAN"  : randint(-4,4),
                        "NECK_TILT" : -1,
                        "NECK_ROLL" : randint(-4,4),
                        "EXPR_SAD": 0.9,
                        "BROW_DOWN_LEFT": 1,
                        "BROW_DOWN_RIGHT": 1
                        
                        }
                    },
                    {
                    "time":[3.32],
                    "persist":False,
                    "params":{
                        "reset":True
                        }
                    }],
                "class":"furhatos.gestures.Gesture"
                },

        ]
   
    }
    
    return gestures