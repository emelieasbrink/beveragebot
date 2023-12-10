from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import os
from read_aus import read_aus_files, calculate_valence, read_both_datsets 


def valence_plot(dataset):
    """
    Creates a plot with calculating the abs difference in AUs from positive and negative valence. 
    The plot is saved in the processed file and it can be used for feature selection.
    Returns the AUs with more than 0.1 in difference. 

    Arguments:
    dataset - ['both', "./processed/Diffusion/cropped/", "./processed/Multi/"] 
    States which dataset to be used for creating the plot (DiffusionFER, MultiEmoVA or both)
    """

    if dataset == 'both':
        df = read_both_datsets()
    elif dataset == "./processed/Diffusion/cropped/":
        df = read_aus_files(dataset)
        df = calculate_valence(df)
    elif dataset == "./processed/Multi/":
        df = read_aus_files(dataset)

    au_columns = [col for col in df.columns if 'AU' in col]
    
    #filt_neg = valence_df['valence'] < 0
    #filt_pos = valence_df['valence'] > 0
    filt_neg = df['label'] == 'negative'
    filt_pos = df['label'] == 'positive'

    columns_to_drop = ['file', 
                       'label',
                       'valence',
                       'face']
    neg_valence = df[filt_neg][au_columns]
    pos_valence = df[filt_pos][au_columns]


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
    if dataset == 'both':
        plt.savefig(Path("./processed/au_visualization.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(Path(dataset + "au_visualization.png"), dpi=300, bbox_inches='tight')
    # Display the plot
    #plt.show()

    features = abs_dif[abs_dif > 0.1]
    return features

if __name__ == "__main__":
    valence_plot('both')
    plt.show()