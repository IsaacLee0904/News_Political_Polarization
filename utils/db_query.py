import pandas as pd

def filter_failed_content(conn, table_name):
    """ 
    Retrieve rows where content length is less than 50 from the specified table 

    Args:
    - conn : Connect to database.
    - table_name (str): Table name in the SQLite database.

    Example usage:
    - conn = create_connection()
      short_content_data = filter_short_content(conn, "nuclear_power")    
    """
    
    query = f"""
            SELECT *
            FROM {table_name} AA
            WHERE 1 = 1
                AND LENGTH(AA.content) < 50
            """
    
    # Execute the query and return the result as a DataFrame
    return pd.read_sql(query, conn)
