# import basic packages
import sys, os, glob, json, re, warnings, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP packages
from ckiptagger import data_utils, WS, POS
# data_utils.download_data_gdown("./ckiptagger/")  # only run if excute first time 
import jieba
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.db_utils import create_connection, get_all_tables_from_db, close_connection
from utils.etl_utils import save_extractdf_to_csv
from utils.nlp_utils import clean_text, tokenize_news_content_with_ckiptagger, tokenize_news_content_with_jieba, extract_content_words, clean_tokens, compute_tfidf, filter_common_words_with_tfidf, tsne_visualization

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure TensorFlow GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():

    ''' setting configure '''
    threshold_value = 0.7
    extract_data_path = os.path.join(project_root, 'data', 'extract_data', 'threshold_{}'.format(threshold_value))

    conn = create_connection()
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
        'nuclear_power': nuclear_power_df.head(100),
        # 'ractopamine': ractopamine_df.head(100),
        # 'alongside_elections': alongside_elections_df.head(100),
        # 'algal_reef': algal_reef_df.head(100)
    }

    for df_key, df_value in final_data.items():
        print(f"Processing {df_key}...")

        # Content cleaning 
        df_value = clean_text(df_value)

        # Tokenize the data
        ws = WS("./ckiptagger")
        pos = POS("./ckiptagger")
        # df_value = tokenize_news_content_with_ckiptagger(df_value, ws)
        df_value = extract_content_words(df_value, ws, pos)

        # Remove stop words
        stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')

        with open(stop_words_path, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]

        df_value = clean_tokens(df_value, stop_words) 

        # TF-IDF
        # Step1. Train the TF-IDF model
        # tfidf_matrix, vectorizer = compute_tfidf(df_value['tokenized_content'].tolist())

        # tsne_visualization(tfidf_matrix, df_value, extract_data_path, df_key)

        # Step2. Use the trained vectorizer to filter out common words   
        # df_value = filter_common_words_with_tfidf(df_value, 'tokenized_content', vectorizer, threshold_value)

        # save extract data to csv
        save_extractdf_to_csv(df_value, extract_data_path, df_key)

if __name__ == "__main__":
    main()