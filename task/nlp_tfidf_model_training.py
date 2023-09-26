# import basic packages
import sys, os, glob, warnings
import csv
import pandas as pd

# NLP packages
from sklearn.feature_extraction.text import TfidfVectorizer

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.log_utils import set_logger
from utils.nlp_utils import clean_tokens
from utils.tf_idf_utils import compute_tfidf
from utils.gpu_utils import check_gpu_availability

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    ''' setting configure '''
    logger = set_logger()
    logger.info("Starting NLP preprocessing : TF-IDF model training...")

    training_data_path = os.path.join(project_root, 'data', 'tokenized_data')
    model_save_path = os.path.join(project_root, 'model', 'tf_idf_model')

    csv_files = glob.glob(os.path.join(training_data_path, '*.csv'))

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

        # Remove stop words
        stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')

        with open(stop_words_path, 'r', encoding='utf-8') as file:
            stop_words = [line.strip() for line in file]

        df_value = clean_tokens(df_value, stop_words, logger) 

        # TF-IDF
        # Step1. Train the TF-IDF model
        df_value = df_value.dropna(subset=['tokenized_content'])
        tfidf_matrix, vectorizer = compute_tfidf(df_value['tokenized_content'].tolist(), model_save_path, df_key, logger)   
        
    logger.info("NLP preprocessing  : TF-IDF model training.")

if __name__ == "__main__":
    main()