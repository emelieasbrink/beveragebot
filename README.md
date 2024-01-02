# beveragebot
Project in course intelligent interactive systems with goal to create a bartender that can hold a conversation with a customer.

To install the required packages run:
`pip install -r requirements. txt` 

To demo the bartender run:
`python main.py` 

**Before running file** Make sure that you set the correct ip adress in the main.py file and the interaction/script.py file. Also start furhat. 

**Note** To be able to run all files in this directory the DiffusionFER dataset and the MultiEmoVA dataset need to be added to the top folder. To demo the bartender this is not needed. 

# Overview project

The project consists of two subsystems: user perception and interaction which have their seperate folders. Integration of the two systems happens in the main.py file. 

## User perception subsystem

**Authors:** Viktoria Svensson, Johanna Dahl and Emelie Åsbrink

The purpose of this system is to process real-time images and classify the facial expression as negative, positive or neutral. A machine learning model has been created for this which is saved in user_perception/model_train/model.joblib.

The main files that creates this model are dataload_aus.py, read_aus.py and training.py. In dataload_aus.py the data is read and processed, creating csv files with action units for each image. This is saved then in the processed folder. To read the data into dataframes the read_aus.py file. This is also where the labels for the classification algorithms are created. The training of different models, saving the one with best accuracy is done in the training.py files.

Some files are not used for creating this final model, but were still important in the model development such as train_landmarks.py where we tested to train a model with landmarks instead as well as model_train/train_gradBoosting.py which trains a gradient boosting classifier which got lower accuracy than other models and takes a long time to run, so it was seperated from the training file. 

## Interaction subsystem

**Authors:** Svante Sundberg and Elsa Strömbäck

# Directory structure
Here you can find a more detailed overview of the directory structure. 

```
├── interaction                     #folder for the interaction subsystem with rule-based system 
    ├── dictionary.py
    ├── gestures.py
    ├── script.py
├── processed                       #folder containing processed image data with action units and landmarks
    ├── Diffusion                   #action units from the DiffusionFER dataset
        ├── cropped                 #processed images from DiffusionFER/cropped
            ├── images              #sample of images with face detection outline
            ├── angry_aus.csv       #csv file with aus values in images with label "angry"
            ├── disgust_aus.csv     #csv file with aus values in images with label "disgust"
            ├── fear_aus.csv        #csv file with aus values in images with label "fear"
            ├── happy_aus.csv       #csv file with aus values in images with label "happy"
            ├── neutral_aus.csv     #csv file with aus values in images with label "neutral"
            ├── sad_aus.csv         #csv file with aus values in images with label "sad"
            ├── surprise_aus.csv    #csv file with aus values in images with label "surprise"
        ├── original                #processed images from DiffusionFER/original
            ├── images              #sample of images with face detection outline
            ├── angry_aus.csv       #csv file with aus values in images with label "angry"
            ├── disgust_aus.csv     #csv file with aus values in images with label "disgust"
            ├── fear_aus.csv        #csv file with aus values in images with label "fear"
            ├── happy_aus.csv       #csv file with aus values in images with label "happy"
            ├── neutral_aus.csv     #csv file with aus values in images with label "neutral"
            ├── sad_aus.csv         #csv file with aus values in images with label "sad"
            ├── surprise_aus.csv    #csv file with aus values in images with label "surprise"
    ├── landmarks                   #sample of images from DiffusionFER/cropped with landmarks outlined
    ├── landmarksCsv                #csv files with landmarks from DiffusionFER/cropped images
    ├── Multi                       #action units from MultiEmoVA dataset
        ├── images                  #sample of images with face detection outline
        ├── negative_aus.csv        #csv file with aus values in images with negative valence
        ├── positive_aus.csv        #csv file with aus values in images with positive valence
        ├── neutral_aus.csv         #csv file with aus values in images with neutral valence
├── user_perception                 #folder for the user perception subsystem 
    ├── model_train                 #contains files for training model with action units
        ├── confusion_matrix.png    #confusion matrix from final 
        ├── feature_sel.py          #creates plot processed/au_vizualisation.py which is the absolute 
                                    value in valence between positive and negative images, 
                                    can be used for feature selection.
        ├── model.joblib            #contains the final model
        ├── pca.joblib              #contains pca transformation for creating features to final model
        ├── read_aus.py             #reads au_files in processed folder into one dataframe and creates labels for 
                                    positive, negative and neutral
        ├── train_gradBoosting.py   #code for gradient boosting classifier, takes long time to run --> 
                                    not included in offical training script. 
                                    Has lower accuracy than other tested models. 
        ├── training.py             #training different models and save the best one in model.joblib file
    ├── dataload_aus.py             #reads MultiEmoVA and DiffusionFER datasets, takes out action units from them and 
                                    saves to processed folder. 
    ├── landmarks.py                #reads DiffsionFER/cropped images and takes out landmarks. 
                                    Saves in processed folder. 
    ├── train_landmarks.py          #training model with landmarks, is not used in the final model
    ├── train.py                    #file for training some different models, is not used in the final model
    ├── video_input.py              #test final model live with webcam 
├── main.py                         #integration of the two subsystems. This is the file to be run for testing the system. 
