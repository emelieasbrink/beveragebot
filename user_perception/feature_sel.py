from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import os

def read_aus_files(folder='cropped'):
    path_aus = "./processed/" + folder + '/'
    all_aus = pd.DataFrame()

    for file in Path(path_aus).iterdir():
        if not file.is_file():
            continue
        if 'csv' in str(file):
            aus_df = pd.read_csv(file)
            all_aus = pd.concat([aus_df, all_aus]).reset_index(drop=True)
    if 'Unnamed: 0' in all_aus.columns:
        all_aus = all_aus.drop(columns='Unnamed: 0')
    return all_aus

def valence_plot(df):
    #read dataset_sheet with info about valence
    path_datasheet = Path("./DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv")
    df_valence = pd.read_csv(path_datasheet)

    df_valence['file_name'] = df_valence['subDirectory_filePath'].apply(lambda x: x.split('/')[-1])
    joined_df = df.merge(df_valence, 
                        left_on = 'file', 
                        right_on='file_name', 
                        validate='one_to_one', 
                        how='inner')
    
    filt_neg = joined_df['valence'] < 0
    filt_pos = joined_df['valence'] > 0

    columns_to_drop = ['subDirectory_filePath', 
                       'arousal', 
                       'expression', 
                       'file', 
                       'label',
                       'valence',
                       'file_name']
    neg_valence = joined_df[filt_neg].drop(columns=columns_to_drop)
    pos_valence = joined_df[filt_pos].drop(columns=columns_to_drop)


    #calculate mean
    mean_pos = neg_valence.mean()
    mean_neg = pos_valence.mean()

    #abs difference between pos and neg
    abs_dif = (mean_pos - mean_neg).abs().sort_values(ascending=False)  

    #plot result
    plt.plot(abs_dif.index, abs_dif.values, marker='o', linestyle='')
    plt.title('Absolute difference in AU means (positive vs. negative)')
    plt.xticks(rotation=45)

    #save fig
    plt.savefig(Path("./processed/cropped/au_visualization.png"), dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()

if __name__ == "__main__":
    df = read_aus_files()
    valence_plot(df)