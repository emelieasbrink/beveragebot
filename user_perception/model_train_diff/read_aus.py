from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

def calculate_valence(df):
    path_datasheet = Path("./DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv")
    df_valence = pd.read_csv(path_datasheet)

    df_valence['file_name'] = df_valence['subDirectory_filePath'].apply(lambda x: x.split('/')[-1])
    joined_df = df.merge(df_valence, 
                        left_on = 'file', 
                        right_on='file_name', 
                        how='inner')
    
    joined_df.rename(columns={"label": "emotion"}, inplace=True)
    conditions = [
        (joined_df['valence'] > 0),
        (joined_df['valence'] < 0),
        (joined_df['valence'] == 0)]
    choices = ['positive', 'negative', 'neutral']

    joined_df['label'] = np.select(conditions, choices)
    columns_to_drop = ['subDirectory_filePath', 
                       'arousal', 
                       'expression', 
                       'file_name']
    joined_df = joined_df.drop(columns=columns_to_drop)
    return joined_df

def read_aus_files(path):
    all_aus = pd.DataFrame()

    for file in Path(path).iterdir():
        if not file.is_file():
            continue
        if 'csv' in str(file):
            aus_df = pd.read_csv(file)
            all_aus = pd.concat([aus_df, all_aus]).reset_index(drop=True)
    return all_aus

def read_both_datsets():
    diff_df = read_aus_files("./processed/Diffusion/original/")
    diff_df = calculate_valence(diff_df)
    multi_df = read_aus_files("./processed/Multi/")
    diff_df.drop(columns='emotion', inplace=True)
    df = pd.concat([diff_df, multi_df]).reset_index(drop=True)
    return df