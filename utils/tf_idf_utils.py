import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(corpus, model_save_path, filename, logger):
    """
    Compute the TF-IDF matrix for a given corpus.

    Parameters:
    - corpus: list
        List of text data to compute TF-IDF.
    - model_save_path: str
        The directory path where the TF-IDF matrix and vectorizer object should be saved.

    Returns:
    - tfidf_matrix: sparse matrix
        The computed TF-IDF matrix.
    - vectorizer: TfidfVectorizer object
        The vectorizer object used for transformation.
    """
    logger.info("Computing TF-IDF matrix for the given corpus.")
    
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        logger.info("Successfully computed the TF-IDF matrix.")
        
        # Save the TF-IDF matrix and vectorizer object to the specified directory
        with open(os.path.join(model_save_path, f'{filename}_tfidf_matrix.pickle'), 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(os.path.join(model_save_path, f'{filename}_vectorizer.pickle'), 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Successfully saved the TF-IDF matrix and vectorizer to {model_save_path}.")
        
    except Exception as e:
        logger.error("Failed to compute the TF-IDF matrix.")
        logger.exception(e)
        raise
    
    logger.info("Number of documents in the corpus: %d", len(corpus))
    logger.info("Number of features (unique words) in the corpus: %d", len(vectorizer.get_feature_names_out()))
    
    return tfidf_matrix, vectorizer

def load_tfidf_objects(model_save_path, filename, logger):
    """
    Load the TF-IDF matrix and the TfidfVectorizer object from the specified directory.
    
    Parameters:
    - model_save_path: str
        The directory path where the TF-IDF matrix and vectorizer object are saved.
    - logger: Logger object
        The logger object used for logging information, warnings, and errors.
        
    Returns:
    - tfidf_matrix: sparse matrix
        The loaded TF-IDF matrix.
    - vectorizer: TfidfVectorizer object
        The loaded vectorizer object used for transformation.
        
    Raises:
    - Exception: If loading the TF-IDF matrix or vectorizer object fails.
    """
    try:
        with open(os.path.join(model_save_path, f'{filename}_tfidf_matrix.pickle'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(os.path.join(model_save_path, f'{filename}_vectorizer.pickle'), 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"Successfully loaded the TF-IDF matrix and vectorizer from {model_save_path}.")
        return tfidf_matrix, vectorizer
    except Exception as e:
        logger.error("Failed to load the TF-IDF matrix and vectorizer.")
        logger.exception(e)
        raise

def filter_common_words_with_tfidf(df, column_name, vectorizer, threshold, logger):
    """
    Filter out common words from a DataFrame based on a pre-trained TF-IDF vectorizer.
    
    This function operates in three main steps:
    1. Filters out words with a length less than 2 from the given column.
    2. Computes the TF-IDF scores for the filtered words using the provided vectorizer.
    3. Filters out the most common words based on the computed TF-IDF scores and the provided threshold.
    
    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the text data to be filtered.
    - column_name: str
        The column name in 'df' that contains the text data to be processed.
    - vectorizer: TfidfVectorizer object
        A pre-trained TF-IDF vectorizer for transforming the text data.
    - threshold: float, default=0.6
        Specifies the proportion of the most common words to filter out based on their TF-IDF scores.
        For example, a threshold of 0.85 means that the top 85% of words, ranked by their TF-IDF scores, will be removed.

    Returns:
    - df: pandas DataFrame
        The DataFrame with the specified column filtered to exclude the common words identified by the threshold.
    """
    logger.info("Starting word filtering...")
    try:
        # Filter out words with length less than MIN_WORD_LENGTH
        MIN_WORD_LENGTH = 2
        df['tokenized_content_TF-IDF'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if len(word) >= MIN_WORD_LENGTH]))
    except Exception as e:
        logger.error(f"Error filtering words with length less than {MIN_WORD_LENGTH}: {e}")

    try:
        # Compute the TF-IDF scores for the filtered words
        tfidf_matrix = vectorizer.transform(df['tokenized_content_TF-IDF'].tolist())
        tfidf_scores = np.sum(tfidf_matrix, axis=0).A1
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        
    except Exception as e:
        logger.error(f"Error computing TF-IDF scores: {e}")

    try:
        # Filter out common words based on the threshold
        logger.info(f"TF-IDF threshold: {threshold}")
        num_words_to_filter = int(len(tfidf_scores) * threshold)
        common_words = set([vectorizer.get_feature_names_out()[idx] for idx in sorted_indices[:num_words_to_filter]])
        
        df['tokenized_content_TF-IDF'] = df['tokenized_content_TF-IDF'].apply(lambda x: ' '.join([word for word in x.split() if word not in common_words]))
    except Exception as e:
        logger.error(f"Error filtering common words with threshold {threshold}: {e}")
    logger.info("Word filtering completed.")

    return df

def filter_tfidf_matrix(tfidf_matrix, vectorizer, common_words, logger):
    """
    Filter out columns from the tfidf_matrix corresponding to common_words.
    """
    # Log the initial shape of the tfidf_matrix
    initial_shape = tfidf_matrix.shape
    logger.info(f"Initial shape of tfidf_matrix: {initial_shape}")
    
    # Get the indices of the common words
    indices = [vectorizer.vocabulary_[word] for word in common_words if word in vectorizer.vocabulary_]
    
    # Log the number of common words found in the vocabulary
    logger.info(f"Number of common words found in the vocabulary: {len(indices)}")
    
    # Check if any common word is not in the vocabulary
    not_in_vocab = [word for word in common_words if word not in vectorizer.vocabulary_]
    if not_in_vocab:
        logger.warning(f"Common words not found in the vocabulary: {not_in_vocab}")
    
    # Remove the columns corresponding to the common words
    filtered_matrix = tfidf_matrix[:, [i for i in range(tfidf_matrix.shape[1]) if i not in indices]]
    
    # Log the shape of the filtered_matrix
    logger.info(f"Shape of filtered_matrix after removing common words: {filtered_matrix.shape}")
    
    return filtered_matrix

def tsne_visualization(tfidf_matrix, df_value, logger):
    """
    Visualize the TF-IDF matrix using t-SNE and save the plot based on news source.
    
    Parameters:
    - tfidf_matrix: array-like, shape (n_samples, n_features)
        The TF-IDF matrix.
    - df_value: pandas DataFrame
        The dataframe containing the source column.
    - folder: str
        The directory where the plot should be saved.
    - df_key: str
        Key to be used for filename prefix.
    """ 
    # Define a color map for different sources
    source_list = {
        'Udn': 'red',
        'Chinatimes': 'blue',
        'Libnews': 'green'
    }

    try:
        # Get the colors for each sample
        colors = [source_list[source] for source in source_list]

        # Calculate the cosine similarity between words
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Dimensionality reduction with t-SNE
        tsne_model = TSNE(n_components=2, random_state=0, init='random')
        low_data = tsne_model.fit_transform(tfidf_matrix.toarray())

        # Visualization
        plt.figure(figsize=(10, 10))
        for source, color in source_list.items():
            plt.scatter(low_data[np.array(source_list) == source, 0], 
                        low_data[np.array(source_list) == source, 1], 
                        c=color, label=source)
        plt.title('t-SNE visualization of TF-IDF matrix')
        plt.legend()

        return plt
    except Exception as e:
        logger.error(f"Error in tsne_visualization: {e}")
        raise

def save_plot(plt_obj, folder, filename_prefix):
    """
    Save the given plot object to a file.
    """
    filename = os.path.join(folder, f"{filename_prefix}_plt.png")
    plt_obj.savefig(filename)
    plt_obj.close()
    print(f"Plot saved to: {filename}")