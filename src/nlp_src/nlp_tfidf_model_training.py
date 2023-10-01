# import basic packages
import sys, os, glob, warnings
import csv
import pandas as pd

# NLP packages
from sklearn.feature_extraction.text import TfidfVectorizer

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from utils.log_utils import set_logger
from utils.nlp_utils import clean_tokens
from utils.tf_idf_utils import compute_tfidf
from utils.gpu_utils import check_gpu_availability

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_stop_words(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def main():
    logger = set_logger()
    logger.info("Starting NLP preprocessing : TF-IDF model training...")

    training_data_path = os.path.join(project_root, 'data', 'tokenized_data')
    model_save_path = os.path.join(project_root, 'model', 'tf_idf_model')

    csv_files = glob.glob(os.path.join(training_data_path, '*.csv'))

    check_gpu_availability(logger)

    stop_words_path = os.path.join(project_root, 'assets', 'stop_words.txt')
    stop_words = load_stop_words(stop_words_path)

    for csv_file in csv_files:
        df_key = os.path.basename(csv_file).replace('.csv', '')
        logger.info(f"Processing {df_key}...")

        csv.field_size_limit(2**30)
        
        df_value = pd.read_csv(csv_file, engine='python', low_memory=True)
        df_value = clean_tokens(df_value, stop_words, logger)

        df_value = df_value.dropna(subset=['tokenized_content'])
        tfidf_matrix, vectorizer = compute_tfidf(df_value['tokenized_content'].tolist(), model_save_path, df_key, logger)

    logger.info("NLP preprocessing : TF-IDF model training completed.")


if __name__ == "__main__":
    main()