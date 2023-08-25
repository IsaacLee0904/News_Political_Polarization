
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
4. Run the script using the command: `udnSpider.py 萊豬 2`.
```bash
python udnSpider.py keyword delaytime
```
5. The script will generate a JSON file with the scraped data.

