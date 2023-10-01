import sys, os
import sqlite3

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.log_utils import set_logger

logger = set_logger()

# path of create database 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
db_path = os.path.join(project_root, 'data', 'db', 'news_political_polarization.sqlite')

# check if the database exist
if not os.path.exists(os.path.dirname(db_path)):
    os.makedirs(os.path.dirname(db_path))
    logger.info(f"Created directory: {os.path.dirname(db_path)}")

# create / connect the database
logger.info(f"Connecting to the database at: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# create table 
table_names = ['nuclear_power', 'ractopamine', 'alongside_elections', 'algal_reef']

for table in table_names:
    logger.info(f"Creating table '{table}' if it doesn't exist.")
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
    logger.info(f"Table '{table}' has been set up.")

conn.commit()
conn.close()
logger.info("Database connection has been closed.")
