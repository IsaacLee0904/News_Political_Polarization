import re, os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ckiptagger import data_utils, WS, POS
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

def clean_text(df):
    """
    Clean the 'content' column of a DataFrame and remove stop words.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be cleaned.

    Returns:
    - df: pandas DataFrame
        A DataFrame with cleaned 'content'.
    """
    punctuations = r'[&#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation 【】"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;《》「」]'
    
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(punctuations, '', x))
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))  # Remove English words
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    return df

def tokenize_news_content_with_ckiptagger(df, ws):
    """
    Tokenize the 'content' column of a DataFrame using CKIPtagger

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be tokenized.
    - ws: CKIPtagger WS object
        The CKIPtagger WS object for tokenization.

    Returns:
    - tokenized_df: pandas DataFrame
        A new DataFrame with an added 'tokenized_content' column.
    """
    df.loc[:, 'tokenized_content'] = df['content'].apply(lambda x: ' '.join(ws([x])[0]))

    return df

def tokenize_news_content_with_jieba(df):
    """
    Tokenize the 'content' column of a DataFrame using jieba.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be tokenized.

    Returns:
    - tokenized_df: pandas DataFrame
        A new DataFrame with an added 'tokenized_content' column.
    """
    
    jieba.initialize()  

    df.loc[:, 'tokenized_content'] = df['content'].apply(lambda x: ' '.join(jieba.cut(x, cut_all=False)))

    return df

def extract_important_terms(text, ws, pos):
    """
    Extract important terms from the text using ckiptagger.
    
    Parameters:
    - text: str
        The input text to be processed.
    - ws: CKIPtagger WS object
        The CKIPtagger WS object for word segmentation.
    - pos: CKIPtagger POS object
        The CKIPtagger POS object for part-of-speech tagging.
        
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

def extract_content_words(df, ws, pos):
    """
    Extract important terms from the 'content' column of a DataFrame using ckiptagger.
    
    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be processed.
    - ws: CKIPtagger WS object
        The CKIPtagger WS object for word segmentation.
    - pos: CKIPtagger POS object
        The CKIPtagger POS object for part-of-speech tagging.
        
    Returns:
    - df: pandas DataFrame
        The DataFrame with an added 'tokenized_content' column containing the extracted important terms.
    """
    
    df['tokenized_content'] = df['content'].apply(lambda x: ' '.join(extract_important_terms(x, ws, pos)))
    return df

def clean_tokens(df, stop_words):
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
    punctuations = r'[&#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation 【】"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;《》「」]'
    df.loc[:, 'tokenized_content'] = df['tokenized_content'].apply(lambda x: re.sub(r'\b\d+\b', '', x))  # Remove standalone numbers 
    df.loc[:, 'tokenized_content'] = df['tokenized_content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    return df

def compute_tfidf(corpus):
    """
    Compute the TF-IDF matrix for a given corpus.

    Parameters:
    - corpus: list
        List of text data to compute TF-IDF.

    Returns:
    - tfidf_matrix: sparse matrix
        The computed TF-IDF matrix.
    - vectorizer: TfidfVectorizer object
        The vectorizer object used for transformation.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix, vectorizer

def filter_common_words_with_tfidf(df, column_name, vectorizer, threshold):
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
    
    # Filter out words with length less than 2
    df['tokenized_content_TF-IDF'] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if len(word) >= 2]))
    
    # Compute the TF-IDF scores for the filtered words
    tfidf_matrix = vectorizer.transform(df['tokenized_content_TF-IDF'].tolist())
    tfidf_scores = np.sum(tfidf_matrix, axis=0).A1
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    
    # Filter out common words based on the threshold
    print('TF-IDF threshold : '+ str(threshold))
    num_words_to_filter = int(len(tfidf_scores) * threshold)
    common_words = set([vectorizer.get_feature_names_out()[idx] for idx in sorted_indices[:num_words_to_filter]])
    
    df['tokenized_content_TF-IDF'] = df['tokenized_content_TF-IDF'].apply(lambda x: ' '.join([word for word in x.split() if word not in common_words]))
    
    return df

def word_frequency_calculation(df):
    
    word_frequencies = df['tokenized_content'].str.split(expand=True).stack().value_counts()

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

def tsne_visualization(tfidf_matrix, df_value, folder, df_key):
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
    colormap = {
        'Udn': 'red',
        'Chinatimes': 'blue',
        'Libnews': 'green'
    }
    
    # Get the colors for each sample
    colors = df_value['source'].map(colormap).tolist()

    # Calculate the cosine similarity between words
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Dimensionality reduction with t-SNE
    tsne_model = TSNE(n_components=2, random_state=0, metric='precomputed', init='random')
    low_data = tsne_model.fit_transform(similarity_matrix)

    # Visualization
    plt.figure(figsize=(10, 10))
    for source, color in colormap.items():
        plt.scatter(low_data[df_value['source'] == source, 0], 
                    low_data[df_value['source'] == source, 1], 
                    c=color, label=source)
    plt.title(f't-SNE visualization of {df_key} TF-IDF matrix')
    plt.legend()

    # Save the plot
    filename = os.path.join(folder, f"{df_key}_plt.png")
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to: {filename}")



