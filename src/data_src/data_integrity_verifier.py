# import basic packages
import sys, os, glob, json, re
import pandas as pd

# import modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from utils.db_utils import create_connection
from utils.db_utils import get_all_tables_from_db
from utils.db_utils import close_connection
from utils.db_query import filter_failed_content
from utils.log_utils import set_logger

logger = set_logger()

logger.info("Script started...")

logger.info("Connecting to the database...")
conn = create_connection()
logger.info("Connected to the database.")

logger.info("Fetching all tables from the database...")
all_data = get_all_tables_from_db(conn)
logger.info(f"Fetched {len(all_data)} tables from the database.")

table_names = ['nuclear_power', 'ractopamine', 'alongside_elections', 'algal_reef']
sources = ['Chinatimes', 'Udn', 'Libnews']

# Create a dictionary to store dataframes
dfs = {}

# Create a list to store results for the table
results = []

for table in table_names:
    logger.info(f"Filtering failed content for table: {table}...")
    data = filter_failed_content(conn, table)
    for source in sources:
        key_name = f"{source.lower()}_{table}"
        dfs[key_name] = data[data['source'] == source]
        
        # Count the number of bad data rows
        count_bad_data = dfs[key_name].shape[0]
        
        # Append results to the list
        results.append([key_name, count_bad_data])

# Print the table
header = ["Table_Source", "Bad_Data_Count"]
print(f"{header[0].ljust(30)} | {header[1].rjust(15)}")
print("-" * 48)
for row in results:
    print(f"{row[0].ljust(30)} | {str(row[1]).rjust(15)}")
    logger.info(f"{row[0].ljust(30)} | {str(row[1]).rjust(15)}")