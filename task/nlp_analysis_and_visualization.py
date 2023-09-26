# import basic packages
import sys, os, glob, json, re, warnings, inspect
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP packages
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.log_utils import set_logger
from utils.db_utils import create_connection, get_all_tables_from_db, close_connection
from utils.etl_utils import save_extractdf_to_csv
from utils.nlp_utils import (
    extract_content_words, 
    clean_tokens, 
    compute_tfidf, 
    filter_common_words_with_tfidf, 
    filter_tfidf_matrix, 
    tsne_visualization,
    save_plot,
    word_frequency_calculation
)
from utils.gpu_utils import check_gpu_availability

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    ''' setting configure '''
    logger = set_logger()
    logger.info("Starting NLP preprocessing : TF-IDF and t-SNE...")

    threshold_value = 0.7
    processing_data_path = os.path.join(project_root, 'data', 'tokenized_data')
    extract_data_path = os.path.join(project_root, 'data', 'extract_data', 'threshold_{}'.format(threshold_value))

    csv_files = glob.glob(os.path.join(processing_data_path, '*.csv'))

    # check the GPU availability
    check_gpu_availability(logger)

    # NLP processing
    for csv_file in csv_files:  
        
        df_key = os.path.basename(csv_file).replace('.csv', '')
        print(f"Processing {df_key}...")
        logger.info(f"Processing {df_key}...")

        # Set the field size limit to a large value
        csv.field_size_limit(2**30)
        
        df_value = pd.read_csv(csv_file, engine='python', low_memory=True)
        print(df_value['tokenized_content'][0]) # for debug

        # Remove stop words
        stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')

        with open(stop_words_path, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]

        df_value = clean_tokens(df_value, stop_words, logger) 

        # TF-IDF
        # Step1. Train the TF-IDF model
        df_value = df_value.dropna(subset=['tokenized_content'])
        df_value = df_value.head(500)
        tfidf_matrix, vectorizer = compute_tfidf(df_value['tokenized_content'].tolist(), logger)   

        # Step2. Use the trained vectorizer to filter out common words   
        df_value = filter_common_words_with_tfidf(df_value, 'tokenized_content', vectorizer, threshold_value, logger)
        print(df_value.head())
        # print(df_value['tokenized_content_TF-IDF'][0]) # for debug
        # common_words = set(df_value['tokenized_content_TF-IDF'].sum().split())

        # # 3. Filter the TF-IDF matrix using the list of common words.
        # filtered_tfidf_matrix = filter_tfidf_matrix(tfidf_matrix, vectorizer, common_words, logger)
        # tsne_pic = tsne_visualization(filtered_tfidf_matrix, df_value)
        # save_plot(tsne_pic, extract_data_path, df_key)

        # # save extract data to csv
        # logger.info(f"Saving extracted data for {df_key} to CSV...")
        # save_extractdf_to_csv(df_value, extract_data_path, df_key, logger)
        # logger.info(f"Data for {df_key} saved successfully.")
        
    logger.info("NLP preprocessing  : TF-IDF and t-SNE completed.")

if __name__ == "__main__":
    main()