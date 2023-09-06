import sys, os, glob, json, re
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.db_utils import create_connection
from utils.db_utils import get_all_tables_from_db
from utils.db_utils import close_connection
from utils.db_query import filter_failed_content

conn = create_connection()
all_data = get_all_tables_from_db(conn)

table_names = ['nuclear_power', 'ractopamine', 'alongside_elections', 'algal_reef']
sources = ['Chinatimes', 'Udn', 'Libnews']

# Create a dictionary to store dataframes
dfs = {}

# Create a list to store results for the table
results = []

for table in table_names:
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



# ---------------- nuclear_power -------------------
nuclear_power_data = filter_failed_content(conn, 'nuclear_power')
chinatimes_nuclear_power = nuclear_power_data[nuclear_power_data['source'] == 'Chinatimes']
udn_nuclear_power = nuclear_power_data[nuclear_power_data['source'] == 'Udn']
libnews_nuclear_power = nuclear_power_data[nuclear_power_data['source'] == 'Libnews']

# ---------------- ractopamine -------------------
ractopamine_data = filter_failed_content(conn, 'ractopamine')
chinatimes_ractopamine = ractopamine_data[ractopamine_data['source'] == 'Chinatimes']
udn_ractopamine = ractopamine_data[ractopamine_data['source'] == 'Udn']
libnews_ractopamine = ractopamine_data[ractopamine_data['source'] == 'Libnews']

# ---------------- ractopamine -------------------
alongside_elections_data = filter_failed_content(conn, 'alongside_elections')
chinatimes_alongside_elections = alongside_elections_data[alongside_elections_data['source'] == 'Chinatimes']
udn_alongside_elections = alongside_elections_data[alongside_elections_data['source'] == 'Udn']
libnews_alongside_elections = alongside_elections_data[alongside_elections_data['source'] == 'Libnews']

# ---------------- algal_reef -------------------
algal_reef_data = filter_failed_content(conn, 'algal_reef')
chinatimes_algal_reef = algal_reef_data[algal_reef_data['source'] == 'Chinatimes']
udn_algal_reef = algal_reef_data[algal_reef_data['source'] == 'Udn']
libnews_algal_reef = algal_reef_data[algal_reef_data['source'] == 'Libnews']

print(udn_nuclear_power['url'][0])
