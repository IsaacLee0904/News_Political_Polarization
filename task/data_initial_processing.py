# import package
import sys, os, glob, json, re
import pandas as pd
import requests
import bs4 as BeautifulSoup

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# import moudle
from utils.etl_utils import get_all_json
from utils.etl_utils import save_csv_to_db
from utils.etl_utils import move_to_backup_folder
from utils.etl_utils import crawl_news
from utils.etl_utils import re_crawl_failed_news

def main():

    ''' setting configure '''
    output_folder = os.path.join(project_root, "data", "raw_data")
    backup_folder = os.path.join(project_root, "data", "backup")
    db_path = os.path.join(project_root, 'data', 'db', 'news_political_polarization.sqlite')

    ''' main workflow '''          
    json_list = get_all_json()  

    for data_json in json_list:

        file_name = '_'.join(os.path.splitext(os.path.basename(data_json))[0].split('_')[-2:])
        print("{0} Loading : {1} {0}".format('-' * 40, file_name))

        # split out news source
        match = re.search(r'_(\w+)_(\w+).json$', data_json)
        if match:
            source = match.group(1)
            keyword = match.group(2)
            print('News source : ' + str(source) + '_' + str(keyword))
        else:
            print("No match found!")
        
        # storage json to csv file
        with open(data_json, 'r' , encoding='utf-8') as read_file:
            lines = read_file.readlines()
            data_list = []
            for line in lines:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Error decoding JSON from line:", line)

            print('Size of data_list:', len(data_list))

            news_df = pd.DataFrame(data_list, columns = ['url' , 'title', 'category', 'up_datetime', 'content'])

            # data cleaning
            news_df['source'] = source 
            news_df['keyword'] = keyword 
            news_df["content"] = news_df["content"].replace('\n', '')
            
            # reshape df 
            news_df = news_df[['source', 'keyword', 'title', 'category', 'up_datetime', 'content', 'url']]
            re_crawl_failed_news(news_df)
            print(news_df.head())
            
            # convert dataframe as csv file
            new_file_name = os.path.join(output_folder, file_name + ".csv")
            news_df.to_csv(new_file_name, index = False)

            # storage dataframe into database
            try:
                save_csv_to_db(new_file_name, db_path)
                move_to_backup_folder(data_json, backup_folder)
                print(f"{data_json} has been moved to backup folder.")
                    
            except Exception as e:
                print(f"Error processing {data_json}: {e}")

if __name__ == '__main__':
    main()