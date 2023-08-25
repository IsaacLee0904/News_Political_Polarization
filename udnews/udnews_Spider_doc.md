
# Selenium Web Scraping Project

This project demonstrates web scraping using Selenium and Scrapy-Selenium. Below, you'll find instructions on setting up your environment and the actual code used for the scraping.

---

## Installation Guide

# Selenium and Scrapy-Selenium Installation Guide

This guide will walk you through the process of setting up Selenium and Scrapy-Selenium for your development environment.

### Prerequisites

Ensure that you have the following installed:
- Python
- pip (Python package installer)

### Steps

1. Install Selenium
To install the Selenium package, run the following command:
```bash
pip install selenium
```

2. Install Scrapy-Selenium
To install the Scrapy-Selenium package, run the following command:
```bash
pip install scrapy-selenium
```

3. Install ChromeDriver
To use Selenium with the Chrome browser, you'll need the ChromeDriver. Here's how to get it:
1. Visit the [ChromeDriver download page](https://chromedriver.storage.googleapis.com/index.html?path=99.0.4844.51/).
2. Download the appropriate version for your system.
3. Extract the downloaded file to a directory of your choice.
4. Add the directory containing the `chromedriver` executable to your system's PATH, or specify its location directly in your code.

---

## Usage

1. Ensure you have followed the installation guide above.
2. Clone this repository to your local machine.
3. Navigate to the directory containing `udnSpider.py`.
4. Run the script using the command: `python udnSpider.py`.
5. The script will generate a JSON file with the scraped data.

---

## Script: udnSpider.py

```python
### imporat package
import datetime
from time import sleep
import sys
import re
## package for web scraping
from bs4 import BeautifulSoup
import requests
import json
## web crawling with Selenium
# basic selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
# Exceptions related to Selenium
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException


def onStart():
    suffix_output = "Udn.json"
    currentDT = datetime.datetime.now()
    dictfilename = currentDT.strftime("%Y%m%d%H%M%S_") + suffix_output
    print('[onStart]: ', dictfilename)
    
    # Open the file with utf-8 encoding
    json_file = open(dictfilename, 'w', encoding='utf-8')
    
    return json_file

def onStop(json_file):
    json_file.close()

def process_item(json_file, item):
    line = json.dumps(dict(item),ensure_ascii=False) + "\n"
    json_file.write(line)

    return item

def get_selenium():                           
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('headless')
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")   
    options.add_experimental_option('prefs', {'profile.default_content_setting_values.notifications': 2})                     
    driver = webdriver.Chrome(options=options)

    return driver

def snapshot_page_source(checkpoint, driver):
    # Storing the page source in page variable
    page = driver.page_source.encode('utf-8')
    # print(page)
  
    # create result.html
    toFile = "snapshot_%s.html"%(checkpoint)
    file_ = open(toFile, 'wb')
  
    # Write the entire page content in result.html
    file_.write(page)
  
    # Closing the file
    file_.close()

def close_popup_if_exists(driver):

    try:
        # Switch to the alert
        alert = driver.switch_to.alert
        alert.dismiss()  # Close the popup
        print("Popup closed.")

    except NoAlertPresentException:
        # No popup found
        pass

def get_level_1(driver, keyword, delay_time):
    '''
    Start to crawl a webpage, and using beautifulsoup
    '''


    start_url = "https://udn.com/search/word/2/%s" % (keyword)

    # Configure Chrome to ignore SSL certificate errors
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument("--ignore-ssl-errors=yes")
    chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})
    # Initialize the driver with the options
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(start_url)

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    counter_i = 0
    per_page = 10
    last_snapshot = 0

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        sleep(delay_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

        # if conditioned, snapshot the page source
        counter_i
        if counter_i > (last_snapshot + per_page):
            last_snapshot = counter_i
            snapshot_page_source(str(last_snapshot), driver)

    # Re-find the elements every time to avoid StaleElementReferenceException
    level_1_list = driver.find_elements(by=By.XPATH, value='//div[@class="story-list__text"]')

    i = 0
    layer1_results = []  # Container to store all the a_item results

    while i < len(level_1_list):
        # Re-fetch the updated list of elements
        sublinks = driver.find_elements(by=By.XPATH, value='//div[@class="story-list__text"]')

        # Check if i is still a valid index for sublinks
        if i >= len(sublinks):
            break

        sublink = sublinks[i]
        a_item = dict()
        
        try:
            node_child = sublink.find_element(by=By.XPATH, value='h2/a')
            a_item['title'] = node_child.text
            a_item['url'] = node_child.get_attribute('href')
        except NoSuchElementException:
            pass

        try:
            node_child = sublink.find_element(by=By.XPATH, value='div[@class="story-list__info"]/a')
            a_item['category'] = node_child.text
        except NoSuchElementException:
            pass

        try:
            node_child = sublink.find_element(by=By.XPATH, value='div[@class="story-list__info"]/time')
            a_item['up_datetime'] = node_child.text
        except NoSuchElementException:
            pass

        # print(a_item)
        layer1_results.append(a_item)  # Add the a_item to the results list
        i += 1  # Increment the iterator

    # driver.quit()

    return layer1_results

def get_level_2(driver, url):
    # using Selenium drive to news URL
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    content_selector = '.container .article-content__paragraph'
    content_divs = soup.select(content_selector)

    # If content_divs is an empty list, change the content_selector
    if not content_divs:
        content_selector = '.container .story__text__wrapper'
        content_divs = soup.select(content_selector)
    
    # get entire content
    full_content = ' '.join([div.text.strip() for div in content_divs])
    full_content = full_content.replace('\n', '')

    return full_content if full_content else 'Content not found'

def main(keyword, delay_time):

    # Initializing the JSON file for output
    json_file = onStart()

    driver = get_selenium()

    layer1_data = get_level_1(driver, keyword, delay_time)

    for news in layer1_data:
        
        try:      
            start_url = news['url']
            content = get_level_2(driver, start_url)
            news['content'] = content
            print(news)
            
            # Using process_item to save the news data to the JSON file
            process_item(json_file, news)

        except:
            pass

    # Closing the JSON file after saving all the news data
    onStop(json_file)


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Usage: %s <keyword> <delay seconds>"%sys.argv[0])
        sys.exit(1)

    if sys.argv[1] and sys.argv[2]:
        main(sys.argv[1], int(sys.argv[2]))
```

---

**Note:** Ensure that you always have the necessary permissions to scrape a website. Always respect `robots.txt` and terms of service of the website.
