import re 
from nltk.corpus import wordnet

def create_keywords_dict():
    # Building a list of Keywords
    list_words=['hello','sweet','sour','strong','beer','wine','cider','goodbye','good','disgusting','joke']
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
    keywords['greet']=[]
    keywords['greet'].append('.*\\b'+'hi'+'\\b.*')
    keywords['greet'].append('.*\\b'+'hey'+'\\b.*')
    # Populating the values in the keywords dictionary with synonyms of keywords formatted with RegEx metacharacters 
    for synonym in list(list_syn['hello']):
        keywords['greet'].append('.*\\b'+synonym+'\\b.*')
    

    # Defining a new key in the keywords dictionary
    keywords['sweet drink']=[]
    keywords['sweet drink'].append('.*\\b'+'fresh'+'\\b.*')
    for synonym in list(list_syn['sweet']):
        keywords['sweet drink'].append('.*\\b'+synonym+'\\b.*')

    keywords['sour drink']=[]
    for synonym in list(list_syn['sour']):
        keywords['sour drink'].append('.*\\b'+synonym+'\\b.*')
    
    keywords['strong drink']=[]       
    keywords['strong drink'].append('.*\\b'+'alcohol'+'\\b.*')
    keywords['strong drink'].append('.*\\b'+'liquor'+'\\b.*')
    for synonym in list(list_syn['strong']):
        keywords['strong drink'].append('.*\\b'+synonym+'\\b.*')

    keywords['beer']=[]
    for synonym in list(list_syn['beer']):
        keywords['beer'].append('.*\\b'+synonym+'\\b.*')

    keywords['wine']=[]
    for synonym in list(list_syn['wine']):
        keywords['wine'].append('.*\\b'+synonym+'\\b.*')
    
 
    keywords['goodbye']=[]
    keywords['goodbye'].append('.*\\b'+'bye'+'\\b.*')
    for synonym in list(list_syn['goodbye']):
        keywords['goodbye'].append('.*\\b'+synonym+'\\b.*')
    
    keywords['good']=[]
    keywords['good'].append('.*\\b'+'yummy'+'\\b.*')
    keywords['good'].append('.*\\b'+'nice'+'\\b.*')
    keywords['good'].append('.*\\b'+'tasty'+'\\b.*')
    keywords['good'].append('.*\\b'+'yes'+'\\b.*')
    keywords['good'].append('.*\\b'+'thank you'+'\\b.*')
    keywords['good'].append('.*\\b'+'thanks'+'\\b.*')
    for synonym in list(list_syn['good']):
        keywords['good'].append('.*\\b'+synonym+'\\b.*')

    
    keywords['disgusting']=[]
    for synonym in list(list_syn['disgusting']):
        keywords['disgusting'].append('.*\\b'+synonym+'\\b.*')
    keywords['disgusting'].append('.*\\b'+'nasty'+'\\b.*')
    keywords['disgusting'].append('.*\\b'+'bad'+'\\b.*')
    keywords['disgusting'].append('.*\\b'+'gross'+'\\b.*')
    keywords['disgusting'].append('.*\\b'+'no'+'\\b.*')
    keywords['disgusting'].append('.*\\b'+'nope'+'\\b.*')

    keywords['joke']=[]
    for synonym in list(list_syn['joke']):
        keywords['joke'].append('.*\\b'+synonym+'\\b.*')


    for intent, keys in keywords.items():
        keywords_dict[intent]=re.compile('|'.join(keys))
    
    keywords['fallback']=[]
    return keywords_dict

def create_responses():
# Building a dictionary of responses
    responses = {
    'greet': {
        'neutral_responses': ['Hello! Welcome to my bar. What kind of drinks do you like?'],
        'positive_responses': ['Hey there! Excited to serve your happy face. Whats your drink of choice'],
        'negative_responses': ['Hi. You look moody. What drink do you prefer?']
    },
    'sweet drink': {
        'neutral_responses': ['Maybe a Mojito?'],
        'positive_responses': ['A Mojito could be perfect for your taste buds!'],
        'negative_responses': ['Have you considered a sweeter option than your mood, like a Mojito?']
        },
    'sour drink': {
        'neutral_responses': ['What do you think about a Lime cocktail?'],
        'positive_responses': ['A Lime cocktail could add a zing to your already positive evening!'],
        'negative_responses': ["You look like you need something strong to bring up the mood, consider a Lime cocktail."]
    },
    'strong drink': {
    'neutral_responses': [
        "Looking for something to kick-start the evening? How about a Long Island Iced Tea?",
        "Something that packs a punch? How about a classic Long Island Iced Tea?"],
    'positive_responses': [
        "Absolutely! How about this powerhouse? A Long Island Iced Tea could be perfect!",
        "You got it! How about diving into a Long Island Iced Tea for that extra kick?"],
    'negative_responses': [
        "I'm afraid I can't recommend something too strong right now. How about a milder option?",
        "I'm not sure a strong drink is the best choice at the moment. Can I suggest something else?"]
    },
    'beer': {
            'neutral_responses': ['I have great beer for you if you want?'],
            'positive_responses': ['Beer it is! You have an excellent taste.'],
            'negative_responses': ['Feeling like a beer on this moody day?']
        },
    'wine': {
            'neutral_responses': ['I have some great wines for you if you want?'],
            'positive_responses': ['Excellent choice! Wine it is.'],
            'negative_responses': ['Maybe wine is good to bring up your mood today.']
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
        'neutral_responses': ['Nice to meet you! Goodbye'],
        'positive_responses': ['You look Happy!Enjoy the bar! Goodbye'],
        'negative_responses': ['Bye you negative person, Come back if you want another drink']
    },
    'good': {
        'neutral_responses': ['Great! Give me a second and I will prepair your drink. Do you want anything else?'],
        'positive_responses': ['Amazing, Keep that smile while I prepair your drink. Do you want anything else?'],
        'negative_responses': ['Okey I will fix the drink staight away then. Do you want anything else? ']
    },
    'disgusting': {
        'neutral_responses': ['Hmm okey. Do you have another drink preference?'],
        'positive_responses': ['Thats to bad, what other drink preference do you have?'],
        'negative_responses': ['Thats a shame.You obviously need a drink. What else do you like?']
    },
    'joke': {
        'positive_responses': [
        "Great! Here's a cheerful one: Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Sure thing! How about this: Why don't we ever see elephants hiding in trees? Because they're so good at it!",
        "Of course, here's a positive one: What did one plate say to the other? Lunch is on me!"],
    'negative_responses': [
        "Hmm, here's a joke to lighten the mood: Why don't we trust stairs? Because they're always up to something!",
        "Alright, a joke might help: Why did the bicycle fall over? Because it was two-tired!",
        "I hope this makes you smile: What do you call fake spaghetti? An impasta!"],
    'neutral_responses': [
        "Sure, here's one: Why don't scientists trust atoms? Because they make up everything!",
        "Okay, here's a classic: Why did the tomato turn red? Because it saw the salad dressing!",
        "Here's a light one: What's orange and sounds like a parrot? A carrot!"]
    }
    }
    return responses

