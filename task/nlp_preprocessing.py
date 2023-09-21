# import basic packages
import sys, os, glob, json, re, warnings, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP packages
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
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
    clean_text, 
    tokenize_news_content_with_ckiptransformers, 
    extract_content_words, 
    clean_tokens, 
    compute_tfidf, 
    filter_common_words_with_tfidf, 
    filter_tfidf_matrix, 
    tsne_visualization,
    save_plot
)

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    ''' setting configure '''
    logger = set_logger()
    logger.info("Starting NLP preprocessing...")

    threshold_value = 0.7
    extract_data_path = os.path.join(project_root, 'data', 'extract_data', 'threshold_{}'.format(threshold_value))

    logger.info("Connecting to the database...")
    conn = create_connection()
    logger.info("Fetching all tables from the database...")
    all_data = get_all_tables_from_db(conn)

    nuclear_power_df = all_data['nuclear_power'] # 核四
    ractopamine_df = all_data['ractopamine'] # 美豬
    alongside_elections_df = all_data['alongside_elections'] # 公投綁大選
    algal_reef_df = all_data['algal_reef'] # 珍愛藻礁

    shapes = {
        'nuclear_power_df': nuclear_power_df.shape,
        'ractopamine_df': ractopamine_df.shape,
        'alongside_elections_df': alongside_elections_df.shape,
        'algal_reef_df': algal_reef_df.shape
    }

    # Format and print the shapes 
    print("---------------- raw_data_shape -------------------")
    for name, shape in shapes.items():
        print(f"{name}: {shape}")
    print('\n')

    # Filter the DataFrames
    filtered_data = {key: df[df['content'].str.len() >= 50] for key, df in all_data.items()}

    # Extract the filtered DataFrames
    nuclear_power_df = filtered_data['nuclear_power']
    ractopamine_df = filtered_data['ractopamine']
    alongside_elections_df = filtered_data['alongside_elections']
    algal_reef_df = filtered_data['algal_reef']

    new_shapes = {
        'nuclear_power_df': nuclear_power_df.shape,
        'ractopamine_df': ractopamine_df.shape,
        'alongside_elections_df': alongside_elections_df.shape,
        'algal_reef_df': algal_reef_df.shape
    }

    # Format and print the shapes 
    print("---------------- clean_data_shape -------------------")
    for name, shape in new_shapes.items():
        print(f"{name}: {shape}")
    print('\n')

    # NLP processing
    final_data = {
        'nuclear_power': nuclear_power_df,
        # 'ractopamine': ractopamine_df,
        # 'alongside_elections': alongside_elections_df,
        # 'algal_reef': algal_reef_df
    }

    for df_key, df_value in final_data.items():
        print(f"Processing {df_key}...")
        logger.info(f"Processing {df_key}...")

        # Content cleaning 
        df_value = clean_text(df_value, logger)

        # Tokenize the data
        ws = WS("./ckiptagger")
        pos = POS("./ckiptagger")
        df_value = tokenize_news_content_with_ckiptagger(df_value, ws, logger)
        
        # Remove stop words
        stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')

        with open(stop_words_path, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]

        df_value = clean_tokens(df_value, stop_words, logger) 

        # TF-IDF
        # Step1. Train the TF-IDF model
        tfidf_matrix, vectorizer = compute_tfidf(df_value['tokenized_content'].tolist(), logger)   

        # Step2. Use the trained vectorizer to filter out common words   
        df_value = filter_common_words_with_tfidf(df_value, 'tokenized_content', vectorizer, threshold_value, logger)
        common_words = set(df_value['tokenized_content_TF-IDF'].sum().split())

        # 3. Filter the TF-IDF matrix using the list of common words.
        filtered_tfidf_matrix = filter_tfidf_matrix(tfidf_matrix, vectorizer, common_words, logger)
        tsne_pic = tsne_visualization(filtered_tfidf_matrix, df_value)
        save_plot(tsne_pic, extract_data_path, df_key)

        # save extract data to csv
        logger.info(f"Saving extracted data for {df_key} to CSV...")
        save_extractdf_to_csv(df_value, extract_data_path, df_key, logger)
        logger.info(f"Data for {df_key} saved successfully.")
        
    logger.info("NLP preprocessing completed.")

if __name__ == "__main__":
    main()