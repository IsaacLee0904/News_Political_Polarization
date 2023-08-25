
# chinatimes Spider Doc.
## Overview

`chinatimesSpider` is a Scrapy spider designed to crawl and extract news articles from `chinatimes.com`. It fetches the URL, title, category, update time, and content of the articles based on a specific search query.

## Prerequisites

- Python 3.x
- Scrapy
- BeautifulSoup4
- requests

## Installation

1. Ensure you have Python 3.x installed. You can verify this by running:
```shell
python --version
```

2. Install the required packages:
```shell
pip install scrapy beautifulsoup4 requests
```

3. Download the `chinatimesSpider.py` script and place it in your Scrapy project's spiders directory.

## Usage

1. Navigate to your Scrapy project's spiders directory.
2. Ensure the `chinatimesSpider.py` script is in this directory.
3. Run the spider with:
```shell
scrapy crawl chinatimesSpider
```

## Functionality

- The spider begins its crawl from a predefined search URL on `chinatimes.com`.
- It extracts the following details from the search results:
  - URL
  - Title
  - Category
  - Update Time
- If a valid article link is found, the spider fetches the content of the article.
- The spider handles pagination and will continue to crawl until no "next" button is found.

## Note

- The spider is currently set to a specific search query. Modify the `start_urls` in the script to adapt to different search queries.
- Ensure `chinatimes.com` allows web crawling as per their `robots.txt` before running the spider.

