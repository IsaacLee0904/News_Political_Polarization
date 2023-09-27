# import basic packages
import sys, os, glob, json, re, warnings, inspect
import pandas as pd
import numpy as np

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.etl_utils import load_csvs_from_directory
from utils.nlp_utils import word_frequency_calculation

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compute_word_frequencies(data):
    return {key: word_frequency_calculation(df) for key, df in data.items()}

def main():

    data_path = os.path.join(project_root, 'data', 'extract_data', 'threshold_0.5')

    data_dict = load_csvs_from_directory(data_path)
    word_frequencies = compute_word_frequencies(data_dict)
    word_frequencies_df = pd.DataFrame.from_dict(word_frequencies['nuclear_power'])
    word_frequencies_df.to_csv(r'fre.csv')
    

if __name__ == "__main__":
    main()