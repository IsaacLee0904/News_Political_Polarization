import re, os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

def clean_text(df, logger):
    """
    Clean the 'content' column of a DataFrame and remove stop words.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be cleaned.

    Returns:
    - df: pandas DataFrame
        A DataFrame with cleaned 'content'.
    """
    try:
        logger.info("Clean content...")
        punctuations = r'[&#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation 【】"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;《》「」]'
        
        df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(punctuations, '', x))
        df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))  # Remove English words
        df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
    
    return df

def tokenize_news_content_with_ckiptransformers(df, ws, logger):
    """
    Tokenize the 'content' column of a DataFrame using CKIP_transformers

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be tokenized.
    - ws: CKIP_transformers WS object
        The CKIP_transformers WS object for tokenization.

    Returns:
    - tokenized_df: pandas DataFrame
        A new DataFrame with an added 'tokenized_content' column.
    """
    try:
        logger.info("Tokenize news...")
        df.loc[:, 'tokenized_content'] = df['content'].apply(lambda x: ' '.join(ws([x])[0]))
    except Exception as e:
        logger.error(f"Error in tokenize_news_content_with_ckiptransformers: {e}")

    return df

def extract_important_terms(text, ws, pos):
    """
    Extract important terms from the text using CKIP_transformers.
    
    Parameters:
    - text: str
        The input text to be processed.
    - ws: CKIP_transformers WS object
        The CKIP_transformers WS object for word segmentation.
    - pos: CKIP_transformers POS object
        The CKIP_transformers POS object for part-of-speech tagging.
        
    Returns:
    - important_terms: list of str
        A list containing the extracted important terms.
    """
    # Perform word segmentation and POS tagging
    word_sentence_list = ws([text])
    pos_sentence_list = pos(word_sentence_list)
    
    # Extract words based on their POS tags
    important_terms = []
    for words, tags in zip(word_sentence_list, pos_sentence_list):
        for word, tag in zip(words, tags):
            if tag in ["N", "Nb", "V", "A"]:
                important_terms.append(word)

    return important_terms

def extract_content_words(df, ws, pos, logger):
    """
    Extract important terms from the 'content' column of a DataFrame using CKIP_transformers.
    
    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be processed.
    - ws: CKIP_transformers WS object
        The CKIP_transformers WS object for word segmentation.
    - pos: CKIP_transformers POS object
        The CKIP_transformers POS object for part-of-speech tagging.
        
    Returns:
    - df: pandas DataFrame
        The DataFrame with an added 'tokenized_content' column containing the extracted important terms.
    """
    try:
        logger.info("Extract important words...")
        df['tokenized_content'] = df['content'].apply(lambda x: ' '.join(extract_important_terms(x, ws, pos)))
    except Exception as e:
        logger.error(f"Error during processing 'content' column with extract_content_words: {e}")

    return df

def clean_tokens(df, stop_words, logger):
    """
    Clean the 'tokenized_content' column of a DataFrame and remove stop words.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'tokenized_content' column to be cleaned.
    - stop_words: list
        List of words to be removed from the content.

    Returns:
    - df: pandas DataFrame
        A DataFrame with cleaned 'tokenized_content'.
    """
    try:
        logger.info("Cleaning tokens...")
        punctuations = r'[&#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation 【】"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;《》「」]'
        df.loc[:, 'tokenized_content'] = df['tokenized_content'].apply(lambda x: re.sub(r'\b\d+\b', '', x))  # Remove standalone numbers 
        df.loc[:, 'tokenized_content'] = df['tokenized_content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    except Exception as e:
        logger.error(f"Error in clean_tokens: {e}")

    return df

def word_frequency_calculation(df, logger):
    
    try:
        logger.info("Starting word frequency calculation...")
        word_frequencies = df['tokenized_content'].str.split(expand=True).stack().value_counts()
    except Exception as e:
        logger.error(f"Error occurred while calculating word frequencies: {e}")

    return word_frequencies

def generate_word_embeddings(corpus, size=100, window=5, min_count=1, workers=4):
    """
    Generate Word2Vec embeddings for a given corpus.

    Parameters:
    - corpus: list
        List of tokenized text data.
    - size: int, optional (default=100)
        Dimensionality of the word vectors.
    - window: int, optional (default=5)
        Maximum distance between the current and predicted word within a sentence.
    - min_count: int, optional (default=1)
        Ignores all words with total frequency lower than this.
    - workers: int, optional (default=4)
        Use these many worker threads to train the model.

    Returns:
    - model: Word2Vec object
        The trained Word2Vec model.
    """
    model = Word2Vec(corpus, size=size, window=window, min_count=min_count, workers=workers)

    return model

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

