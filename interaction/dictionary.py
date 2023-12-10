import re 
from nltk.corpus import wordnet

def create_keywords_dict():
    # Building a list of Keywords
    list_words=['hello','sweet','sour','goodbye']
    list_syn={}
    for word in list_words:
        synonyms=[]
        for syn in wordnet.synsets(word):
            for lem in syn.lemmas():
                # Remove any special characters from synonym strings
                lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())
                synonyms.append(lem_name)
        list_syn[word]=set(synonyms)

    keywords={}
    keywords_dict={}
    # Defining a new key in the keywords dictionary
    keywords['greet']=[]
    # Populating the values in the keywords dictionary with synonyms of keywords formatted with RegEx metacharacters 
    for synonym in list(list_syn['hello']):
        keywords['greet'].append('.*\\b'+synonym+'\\b.*')

    # Defining a new key in the keywords dictionary
    keywords['sweet drink']=[]
    # Populating the values in the keywords dictionary with synonyms of keywords formatted with RegEx metacharacters 
    for synonym in list(list_syn['sweet']):
        keywords['sweet drink'].append('.*\\b'+synonym+'\\b.*')

    keywords['sour drink']=[]
    # Populating the values in the keywords dictionary with synonyms of keywords formatted with RegEx metacharacters 
    for synonym in list(list_syn['sour']):
        keywords['sour drink'].append('.*\\b'+synonym+'\\b.*')
    
    keywords['goodbye']=[]
    # Populating the values in the keywords dictionary with synonyms of keywords formatted with RegEx metacharacters 
    for synonym in list(list_syn['goodbye']):
        keywords['goodbye'].append('.*\\b'+synonym+'\\b.*')


    for intent, keys in keywords.items():
        keywords_dict[intent]=re.compile('|'.join(keys))

    keywords['fallback']=[]
    return keywords_dict

def create_responses():
# Building a dictionary of responses
    responses = {
    'greet': {
        'natural_responses': ['Hello! Welcome to my bar. What kind of drinks do you like?'],
        'positive_responses': ['Hey there! Excited to serve your happy face. Whats your drink of choice'],
        'negative_responses': ['Hi. You look moody. If youre unsure, I can recommend something. What do you prefer']
    },
    'sweet drink': {
        'natural_responses': ['Maybe a Mojito?'],
        'positive_responses': ['A Mojito could be perfect for your taste buds!'],
        'negative_responses': ['Have you considered a sweeter option than your mood, like a Mojito?']
        },
    'sour drink': {
        'natural_responses': ['What do you think about a Lime cocktail?'],
        'positive_responses': ['A Lime cocktail could add a zing to your already positive evening!'],
        'negative_responses': ["You look like you need something strong to bring up the mood, consider a Lime cocktail."]
    },
    'fallback': {
         'fallback_responses': [
              'I dont quite understand. Could you repeat that?',
              'Sorry, can you specify?',
              'I dont understand, can you repeat?',
              'I beg your pardon, please come again',
              'Please repeat what you said'
        ]
    },
    'goodbye': {
        'natural_responses': ['Nice to meet you! Goodbye'],
        'positive_responses': ['You look Happy!Enjoy the bar! Goodbye'],
        'negative_responses': ['Bye you negative person, Come back if you want another drink']
    }
    }
    return responses