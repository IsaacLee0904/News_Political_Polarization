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

nuclear_power_df = all_data['nuclear_power']
sqlite_sequence_df = all_data['sqlite_sequence']
alongside_elections_df = all_data['alongside_elections']
algal_reef_df = all_data['algal_reef']

print(nuclear_power_df)