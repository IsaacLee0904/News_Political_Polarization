# import package
import sys, os, glob, json, re
import sqlite3
import pandas as pd
import requests
import bs4 as BeautifulSoup
import shutil

def get_all_json():  
    ''' collect all spider file from tHree news folder '''
    main_directory = os.getcwd()  
    
    json_files = glob.glob(os.path.join(main_directory, "crawlers", "*", "*.json"))
    print(json_files)
    json_list = []

    for i in json_files:
        json_list.append(i)

    return json_list

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

def move_to_backup_folder(file_path, backup_folder):
    
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    shutil.move(file_path, os.path.join(backup_folder, os.path.basename(file_path)))