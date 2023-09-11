# import basic packages
import sys, os, glob, json, re, warnings, inspect
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.etl_utils import load_csvs_from_directory
from utils.nlp_utils import word_frequency_calculation

# Configuration settings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compute_word_embeddings(sentences):
    # Train a Word2Vec model
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model

def build_similarity_network(model, terms):
    G = nx.Graph()
    edge_data = []
    
    for term1, term2 in itertools.combinations(terms, 2):
        if term1 in model.wv and term2 in model.wv:
            similarity = cosine_similarity([model.wv[term1]], [model.wv[term2]])[0][0]
            if similarity > 0.5:  # Setting a threshold
                G.add_edge(term1, term2, weight=similarity)
                edge_data.append((term1, term2, similarity))
    
    edge_df = pd.DataFrame(edge_data, columns=['Node1', 'Node2', 'Similarity'])
    edge_df.to_csv("network_data.csv", index=False)
    print("Network data saved to network_data.csv")
    
    return G

def visualize_network_graph(G, term_frequency, df, source_column='source'):
    color_map = {'Udn': 'red', 'Chinatimes': 'blue', 'Libnews': 'green'}

    node_colors = []
    for node in G.nodes():
        source = df[df['tokenized_content'].str.contains(node)][source_column].mode()[0]
        node_colors.append(color_map.get(source, 'black'))

    sizes = [term_frequency[node] * 100 for node in G.nodes()]  # Adjust the multiplier for desired node sizes
    nx.draw(G, with_labels=True, node_size=sizes, node_color=node_colors)
    # plt.show()

    # Save the plot
    # filename = os.path.join(folder, f"{df_key}_plt.png")
    plt.savefig(f"test_plt.png")
    plt.close()

    print(f"Plot saved to: {filename}")

# Load your data
data_path = os.path.join(project_root, 'data', 'extract_data', 'threshold_0.5', 'nuclear_power.csv')
print(data_path)
# Before processing the sentences, check and handle non-string entries
df = pd.read_csv(data_path)
df = df[df['tokenized_content'].notna()]  # Remove rows where 'tokenized_content' is NaN
df['tokenized_content'] = df['tokenized_content'].apply(lambda x: str(x))  # Convert all entries to string
sentences = df['tokenized_content'].str.split().tolist()

# Compute term frequency
all_terms = list(itertools.chain(*sentences))
term_frequency = Counter(all_terms)

# Filter terms for those with a decent frequency for better visualization
filtered_terms = [term for term, freq in term_frequency.items() if freq > 5]

# Compute word embeddings
model = compute_word_embeddings(sentences)

# Build similarity network
G = build_similarity_network(model, filtered_terms)

# Visualize the network graph
visualize_network_graph(G, term_frequency, df)
