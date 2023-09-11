# import package
import sys, os, glob, json, re
import inspect
import shutil
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def get_all_json():  
    ''' collect all spider file from tHree news folder '''
    main_directory = os.getcwd()  
    
    json_files = glob.glob(os.path.join(main_directory, "crawlers", "*", "*.json"))
    print(json_files)
    json_list = []

    for i in json_files:
        json_list.append(i)

    return json_list

import pandas as pd
import os

def load_csvs_from_directory(directory_path):
    """
    Load all CSV files from the specified directory into a dictionary of DataFrames.
    
    Parameters:
    - directory_path: str
        The path to the directory containing the CSV files.
        
    Returns:
    - dict of pandas DataFrame
        A dictionary where the keys are the filenames (without .csv) 
        and the values are the corresponding DataFrames.

    Example usage:
    - dfs = load_csvs_from_directory(directory_path)
    """
    
    dataframes = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            # Remove the .csv extension from the filename to use as the key
            key = filename[:-4]
            filepath = os.path.join(directory_path, filename)
            
            # Read the CSV file into a DataFrame and store in the dictionary
            dataframes[key] = pd.read_csv(filepath)
    
    return dataframes

def save_csv_to_db(csv_filename, db_path):
    """
    Save data from CSV to SQLite table based on the filename format.

    Args:
    - csv_filename (str): Path to the CSV file.
    - db_path (str): Path to the SQLite database.

    Example usage:
    - save_csv_to_db("data_xx_topic.csv", "data/db/mydatabase.sqlite")

    """
    # Extract the 'topic' part from the filename
    topic = os.path.basename(csv_filename).split('_')[1].split('.')[0]
    
    # Determine the table name based on 'topic'
    table_mapping = {
        "萊豬": "ractopamine",
        "藻礁": "algal_reef",
        "公投綁大選": "alongside_elections",
        "核四": "nuclear_power"
    }

    table_name = table_mapping.get(topic)
    
    if not table_name:
        raise ValueError(f"Unrecognized filename format: {csv_filename}")
    
    # Load CSV data
    data = pd.read_csv(csv_filename)
    
    # Save to SQLite
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

import inspect

def save_extractdf_to_csv(df, path, filename):
    """
    Save the given DataFrame to a specified CSV file.
    
    Parameters:
    - df: pandas DataFrame
        The DataFrame to be saved.
    - path: str
        The directory path where the CSV file should be saved.
    - filename: str
        The base name of the CSV file without the extension.
    
    Returns:
    - None
    """
    # Ensure the filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    csv_filepath = os.path.join(path, filename)
    df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f"DataFrame saved to: {csv_filepath}")


def move_to_backup_folder(file_path, backup_folder):
    
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    shutil.move(file_path, os.path.join(backup_folder, os.path.basename(file_path)))

def crawl_news(url):
    """ Retrieve news content from the given URL """
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    container = soup.find(id='container')
    main = container.find('main') if container else None
    if main:
        paragraphs = main.find_all('p')
        content = ' '.join([para.text.strip() for para in paragraphs])
        return content
    else:
        return 'Content not found'

def re_crawl_failed_news(df, delay = 5):
    """ Re-crawl news with missing or very short content """
    
    # Identify rows where content length is less than 50 or content is 'Content not found'
    filter_condition = (df['content'] == 'Content not found')
    
    for idx, row in df[filter_condition].iterrows():
        new_content = crawl_news(row['url'])
        if new_content:
            df.at[idx, 'content'] = new_content
        else:
            df.at[idx, 'content'] = None

    time.sleep(delay)
            
    return df