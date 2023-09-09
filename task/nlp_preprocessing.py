# import basic packages
import sys, os, glob, json, re, warnings
import pandas as pd
import numpy as np

# NLP packages
from ckiptagger import data_utils, WS
# data_utils.download_data_gdown("./ckiptagger/")  # only run if excute first time 
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.db_utils import create_connection, get_all_tables_from_db, close_connection
from utils.nlp_utils import clean_text, tokenize_news_content, compute_tfidf, filter_common_words_with_tfidf, generate_word_embeddings

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

    # Remove stop words
    stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')

    with open(stop_words_path, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file]

    nuclear_power_df = clean_text(nuclear_power_df.head(100), stop_words) # testing with 100 rows

    # Tokenize the data
    ws = WS("./ckiptagger")
    nuclear_power_df = tokenize_news_content(nuclear_power_df, ws) 
    # ractopamine_df = tokenize_news_content(ractopamine_df, ws)
    # alongside_elections_df = tokenize_news_content(alongside_elections_df, ws)
    # algal_reef_df = tokenize_news_content(algal_reef_df, ws)
    print('[Before TF-IDF]')
    print(nuclear_power_df['tokenized_content'][0])
    print('-'*144)

    # TF-IDF
    # Step1. Train the TF-IDF model
    tfidf_matrix, vectorizer = compute_tfidf(nuclear_power_df['tokenized_content'].tolist())

    # Step2. Use the trained vectorizer to filter out common words
    print('[After TF-IDF]')
    nuclear_power_df = filter_common_words_with_tfidf(nuclear_power_df, 'tokenized_content', vectorizer)
    print(nuclear_power_df['tokenized_content'][0])

    # Word embeddings

if __name__ == "__main__":
    main()