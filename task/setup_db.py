import sys, os
import sqlite3

# path of create database 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
db_path = os.path.join(project_root, 'data', 'db', 'news_political_polarization.sqlite')

# check if the database exist
if not os.path.exists(os.path.dirname(db_path)):
    os.makedirs(os.path.dirname(db_path))

# create / connect the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# create table 
table_names = ['nuclear_power', 'ractopamine', 'alongside_elections', 'algal_reef']

for table in table_names:
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            keyword TEXT,
            title TEXT,
            category TEXT,
            up_datetime DATETIME,
            content TEXT,
            url TEXT
        )
    """)

conn.commit()
conn.close()
