![banner](https://cambridge-intelligence.com/wp-content/uploads/2021/05/poster.jpg)

![Python version](https://img.shields.io/badge/Python%20Version-3.9+-lightgrey)
![GitHub last commit](https://img.shields.io/badge/last%20commit-Jul-green)
![GitHub last commit](https://img.shields.io/badge/Repo%20Size-210M-blue)
![GitHub last commit](https://img.shields.io/badge/Project%20Type-Analytical%20Project-red)

Badge [source](https://shields.io/)

# News Political Polarization
This project aims to analyze news content by converting it into a text network, with the goal of demonstrating the phenomenon of political polarization across different media.

## Authors 
- [@IsaacLee0904](https://github.com/IsaacLee0904)

## Table of Contents
  - [Data Info](#Data-Info)
  - [Repository structure](#repository-structure)
  - [News Keyword](#news-keyword)

## Data Info
  - Data Type : Web crawler 
  - Data Source
    * [China Times](https://www.chinatimes.com/?chdtv)
    * [Liberty Times News](https://www.ltn.com.tw/)
    * [United Daily News](https://udn.com/news/index)
  - Keyword
    * 萊豬
    * 藻礁
    * 公投綁大選
    * 核四
 
## Repository structure
```
├── Dockerfile
├── LICENSE
├── analysis
│   ├── nlp_analysis.py
│   └── test.py
├── assets
│   ├── setup_db_flow.jpg
│   └── stop_words.txt
├── crawlers
│   ├── chinatimes
│   │   ├── chinatimes
│   │   │   ├── __init__.py
│   │   │   ├── items.py
│   │   │   ├── middlewares.py
│   │   │   ├── pipelines.py
│   │   │   ├── settings.py
│   │   │   └── spiders
│   │   │       ├── __init__.py
│   │   │       └── chinatimesSpider.py
│   │   ├── chinatimes_spider_doc.md
│   │   └── scrapy.cfg
│   ├── libnews
│   │   ├── libnews
│   │   │   ├── __init__.py
│   │   │   ├── items.py
│   │   │   ├── middlewares.py
│   │   │   ├── pipelines.py
│   │   │   ├── settings.py
│   │   │   └── spiders
│   │   │       ├── __init__.py
│   │   │       └── libnewsSpider.py
│   │   ├── libnews_spider_doc.md
│   │   └── scrapy.cfg
│   └── udnews
│       ├── udnSpider.py
│       └── udnews_Spider_doc.md
├── data
│   ├── backup
│   │   ├── 20230825145838_Chinatimes_萊豬.json
│   │   ├── 20230825185848_Libnews_萊豬.json
│   │   ├── 20230829111408_Udn_萊豬.json
│   │   ├── 20230829133811_Udn_藻礁.json
│   │   ├── 20230829134139_Libnews_藻礁.json
│   │   ├── 20230829165101_Chinatimes_藻礁.json
│   │   ├── 20230830121006_Udn_公投綁大選.json
│   │   ├── 20230830121148_Chinatimes_公投綁大選.json
│   │   ├── 20230830121156_Libnews_公投綁大選.json
│   │   ├── 20230830173313_Udn_核四.json
│   │   ├── 20230830173634_Chinatimes_核四.json
│   │   └── 20230830181537_Libnews_核四.json
│   ├── db
│   │   └── news_political_polarization.sqlite
│   ├── extract_data
│   │   ├── t-SNE_full_result
│   │   │   ├── algal_reef_plt.png
│   │   │   ├── alongside_elections_plt.png
│   │   │   ├── nuclear_power_plt.png
│   │   │   └── ractopamine_plt.png
│   │   ├── threshold_0.3
│   │   │   ├── algal_reef.csv
│   │   │   ├── alongside_elections.csv
│   │   │   ├── nuclear_power.csv
│   │   │   └── ractopamine.csv
│   │   ├── threshold_0.5
│   │   │   ├── algal_reef.csv
│   │   │   ├── alongside_elections.csv
│   │   │   ├── nuclear_power.csv
│   │   │   └── ractopamine.csv
│   │   ├── threshold_0.6
│   │   │   ├── algal_reef.csv
│   │   │   ├── alongside_elections.csv
│   │   │   ├── nuclear_power.csv
│   │   │   └── ractopamine.csv
│   │   └── threshold_0.7
│   │       ├── algal_reef.csv
│   │       ├── alongside_elections.csv
│   │       ├── nuclear_power.csv
│   │       └── ractopamine.csv
│   ├── raw_data
│   │   ├── Chinatimes_核四.csv
│   │   ├── Chinatimes_萊豬.csv
│   │   ├── Chinatimes_藻礁.csv
│   │   ├── Chinatimes_公投綁大選.csv
│   │   ├── Libnews_核四.csv
│   │   ├── Libnews_萊豬.csv
│   │   ├── Libnews_藻礁.csv
│   │   ├── Libnews_公投綁大選.csv
│   │   ├── Udn_核四.csv
│   │   ├── Udn_萊豬.csv
│   │   ├── Udn_藻礁.csv
│   │   └── Udn_公投綁大選.csv
│   └── tokenized_data
│       ├── algal_reef.csv
│       ├── alongside_elections.csv
│       ├── nuclear_power.csv
│       └── ractopamine.csv
├── docs
├── model
│   ├── sentence_transformer_model
│   └── tf_idf_model
│       ├── algal_reef
│       │   ├── algal_reef_tfidf_matrix.pickle
│       │   └── algal_reef_vectorizer.pickle
│       ├── alongside_elections
│       │   ├── alongside_elections_tfidf_matrix.pickle
│       │   └── alongside_elections_vectorizer.pickle
│       ├── nuclear_power
│       │   ├── nuclear_power_tfidf_matrix.pickle
│       │   └── nuclear_power_vectorizer.pickle
│       └── ractopamine
│           ├── ractopamine_tfidf_matrix.pickle
│           └── ractopamine_vectorizer.pickle
├── readme.md
├── reference
├── reports
├── setup.py
├── src
│   ├── __init__.py
│   ├── data_initial_processing.py
│   ├── data_integrity_verifier.py
│   ├── gpu_inspector.py
│   ├── nlp_analysis_and_visualization.py
│   ├── nlp_tfidf_model_training.py
│   ├── nlp_tokenization.py
│   ├── setup_db.py
│   └── task_doc.md
└── utils
    ├── __init__.py
    ├── db_query.py
    ├── db_utils.py
    ├── etl_utils.py
    ├── gpu_utils.py
    ├── log_utils.py
    ├── network_utils.py
    ├── nlp_utils.py
    └── tf_idf_utils.py
```

## Methods
* Exploratory data analysis
* Text Mining
* Natural Language Processing / NLP
* TF-IDF
* Network analysis

## Tech Stack
* Python
* Scrapy
* Selenium
* Ckiptagger




