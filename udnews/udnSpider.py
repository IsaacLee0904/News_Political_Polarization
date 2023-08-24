### imporat package
import datetime
from time import sleep
import sys
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

def get_level_2(driver, url):
    # using Selenium drive to news URL
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    content_selector = '.container .article-content__paragraph'
    content_divs = soup.select(content_selector)
    
    # get entire content
    full_content = ' '.join([div.text.strip() for div in content_divs])
    
    return full_content if full_content else 'Content not found'

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

def main(keyword, delay_time):
    '''
    Start to crawl a webpage, and using beautifulsoup
    '''

    data_json = onStart()

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
    while i < len(level_1_list):
        # Re-fetch the updated list of elements
        sublinks = driver.find_elements(by=By.XPATH, value='//div[@class="story-list__text"]')

        # Check if i is still a valid index for sublinks
        if i >= len(sublinks):
            break

        sublink = sublinks[i]

        try:
            node_child = sublink.find_element(by=By.XPATH, value='h2/a')
            a_item = dict()
            if node_child:
                a_item['title'] = node_child.text
                a_item['url'] = node_child.get_attribute('href')
            node_child = sublink.find_element(by=By.XPATH, value='div[@class="story-list__info"]/a')
            if node_child:
                a_item['category'] = node_child.text
            node_child = sublink.find_element(by=By.XPATH, value='div[@class="story-list__info"]/time')
            if node_child:
                a_item['up_datetime'] = node_child.text
            if a_item:
                a_item['content'] = get_level_2(driver, a_item['url'])
                print('Get a record ===>', a_item)
                process_item(data_json, a_item)

        except NoSuchElementException:
            continue

    

    driver.quit()


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Usage: %s <keyword> <delay seconds>"%sys.argv[0])
        sys.exit(1)

    if sys.argv[1] and sys.argv[2]:
        main(sys.argv[1], int(sys.argv[2]))
