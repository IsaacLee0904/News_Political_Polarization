import sys, os
import pandas as pd
import sqlite3

# path of create database 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
db_path = os.path.join(project_root, 'data', 'db', 'news_political_polarization.sqlite')

def create_connection():
    """ Establish a connection to the SQLite database """
    conn = sqlite3.connect(db_path)
    return conn

def get_all_tables_from_db(conn):
    """ Fetch all data from SQLite and print available tables """
    # Get the list of all tables in the database
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = conn.execute(tables_query).fetchall()
    tables = [table[0] for table in tables]
    
    # Print available tables
    print("Available tables in the database:")
    for table in tables:
        print(f"- {table}")
    
    # Fetch data from each table and store in a dictionary
    dataframes = {}
    for table in tables:
        dataframes[table] = pd.read_sql(f"SELECT * FROM {table};", conn)
    
    return dataframes

def close_connection(conn):
    """ Close the provided database connection """
    conn.close()