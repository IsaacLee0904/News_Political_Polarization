# import package
import sys, os, glob, json, re
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.db_utils import create_connection
from utils.db_utils import get_all_tables_from_db
from utils.db_utils import close_connection

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