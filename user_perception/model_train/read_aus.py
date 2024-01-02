from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

def calculate_valence(df):
    """
    Creates a column label in the dataframe used from training based on the valence of the picture
    The valence is read from the dataset_sheet file and using the following logic the labels are created:
    valence > 0.2: positive
    valence < 0: negative
    valence >= 0 and valence <= 0.2 neutral
    """

    path_datasheet = Path("./DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv")
    df_valence = pd.read_csv(path_datasheet)

    df_valence['file_name'] = df_valence['subDirectory_filePath'].apply(lambda x: x.split('/')[-1])
    joined_df = df.merge(df_valence, 
                        left_on = 'file', 
                        right_on='file_name', 
                        how='inner')
    
    joined_df.rename(columns={"label": "emotion"}, inplace=True)

    #create label positive, negative or neutral from valence score
    conditions = [
        (joined_df['valence'] > 0.2), #pos.
        (joined_df['valence'] < 0), #neg. Changed bc showed almost no negative labels
        (joined_df['valence'] >= 0) & (joined_df['valence'] <= 0.2)] #neutral 

    choices = ['positive', 'negative', 'neutral']

    joined_df['label'] = np.select(conditions, choices)
    #print how many images have each label
    print('label dist', joined_df['label'].value_counts())
    columns_to_drop = ['subDirectory_filePath', 
                       'arousal', 
                       'expression', 
                       'file_name']
    joined_df = joined_df.drop(columns=columns_to_drop)
    return joined_df

def read_aus_files(path):
    """
    Reads the csv files in the specified path and combines them. Returns a dataframe with all of the csv files in one.
    The argument path takes one of the following values ["./processed/Diffusion/original/", ./processed/Diffusion/cropped/, ./processed/Multi/]
    """
    all_aus = pd.DataFrame()

    #iterate through all files in path and save into one dataframe
    for file in Path(path).iterdir():
        if not file.is_file():
            continue
        if 'csv' in str(file):
            aus_df = pd.read_csv(file)
            all_aus = pd.concat([aus_df, all_aus]).reset_index(drop=True)
    return all_aus

def read_both_datsets():
    """Reads the processed files from both dataset DiffusionFER and MultiEmoVA and combines them into one dataframe which is returned"""
    diff_df = read_aus_files("./processed/Diffusion/original/")
    diff_df = calculate_valence(diff_df)
    multi_df = read_aus_files("./processed/Multi/")
    diff_df.drop(columns='emotion', inplace=True)
    df = pd.concat([diff_df, multi_df]).reset_index(drop=True)
    return df

