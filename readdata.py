# import package
import sys, os, glob, json, re
import pandas as pd

def get_all_json():  
    ''' collect all spider file from tHree news folder '''
    main_directory = os.getcwd()  
    
    json_files = glob.glob(os.path.join(main_directory, "*", "*.json"))
    print(json_files)
    json_list = []

    for i in json_files:
        json_list.append(i)

    return json_list

def main(data_json):
    ''' Read collected news as a dict objects list '''
    for data_json in json_list:

        # split out news source
        match = re.search(r'_(\w+).json$', data_json)
        if match:
            source = match.group(1)
            print('News source : ' + str(source))
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

            print(data_list)
            print('Size of data_list:', len(data_list))

            news_df = pd.DataFrame(data_list, columns = ['url' , 'title', 'category', 'up_datetime', 'content'])
            news_df['source'] = source # add new col to the dataframe
            
            # save dataframe as csv file
            base_name = os.path.basename(data_json)
            new_file_name = os.path.splitext(base_name)[0] + ".csv"
            news_df.to_csv(new_file_name)

''' main workflow '''          
json_list = get_all_json()
main(json_list)

