import re
from ckiptagger import data_utils, WS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def clean_text(df, stop_words):
    """
    Clean the 'content' column of a DataFrame and remove stop words.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the 'content' column to be cleaned.
    - stop_words: list
        List of words to be removed from the content.

    Returns:
    - df: pandas DataFrame
        A DataFrame with cleaned 'content'.
    """
    punctuations = r'[&#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation &#8203;``oaicite:{"number":1,"invalid_reason":"Malformed citation 【】"}``&#8203;"}``&#8203;"}``&#8203;"}``&#8203;《》「」]'
    
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(punctuations, '', x))
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))  # Remove English words
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'(?<![^\x00-\x7F])\d+(?![^\x00-\x7F])', '', x))  # Remove standalone numbers but not numbers adjacent to Chinese characters
    df.loc[:, 'content'] = df['content'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    df.loc[:, 'content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    return df

def tokenize_news_content(df, ws):
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

    # Tokenize the 'content' column 
    df.loc[:, 'tokenized_content'] = df['content'].apply(lambda x: ' '.join(ws([x])[0]))

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

def filter_common_words_with_tfidf(df, column_name):
    """
    Filter out common words from a DataFrame based on TF-IDF.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the column to be filtered.
    - column_name: str
        The column name in df to be filtered.

    Returns:
    - df: pandas DataFrame
        A DataFrame with common words filtered out.
    """
    # Compute TF-IDF
    tfidf_matrix, vectorizer = compute_tfidf(df[column_name].tolist())

    # Check if vectorizer has the get_feature_names_out method
    if hasattr(vectorizer, "get_feature_names_out"):
        feature_names = vectorizer.get_feature_names_out()
    else:
        feature_names = vectorizer.get_feature_names()

    # Find words with low TF-IDF score 
    threshold = 0.9
    low_tfidf_words = [word for word, score in zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0]) if score < threshold]

    # Filter out the common words
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in low_tfidf_words]))

    return df

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
