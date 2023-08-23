# import package
import scrapy
import requests
from bs4 import BeautifulSoup

class LibnewsspiderSpider(scrapy.Spider):

    ''' basic info & setting '''
    # named the spider
    name = 'libnewsSpider'
    # define the domains that are allowed to be crawled
    allowed_domains = ['ltn.com.tw']
    ## URL format for China Times search
    # 1. Keyword (necessary)
    # 2. start_time & end_time
    # 3. type 
    # 4. Sorted by news type
    start_urls = ['https://search.ltn.com.tw/list?keyword=萊豬&start_time=20041201&end_time=20230809&sort=date&type=all&sort=date']

    def parse(self, response):
        print( '[LibnewsspiderSpider] Enter parse:', response, type(response), dir(response) )
        #full xpath: /html/body/section/div[6]/ul/li
        #xpath: //*[@id="ec"]/section/div[6]/ul/li
        level_1_list = response.xpath('//*[@id="ec"]/section/div[6]/ul/li')              # this will be get a generator

        for sublink in level_1_list:
            URL = sublink.xpath('a/@href' ).get()       # 此選擇節點下，找到的第一個 <a href=""...> tag
            title = sublink.xpath('a/@title' ).get()    # 此選擇節點下，找到的第一個 <a title=""...> tag
            category = sublink.xpath('div/i/text()' ).get()    # 此選擇節點下，找到的第一個 <div...><i ...> tag         //*[@id="ec"]/section/div[6]/ul/li[1]/div/i
            up_datetime = sublink.xpath('div/span/text()' ).get()  # 此選擇節點下，找到的第一個 <div...><span ...> tag  //*[@id="ec"]/section/div[6]/ul/li[1]/div/span

            a_item = { 'url': URL, 'title': title, 'category': category, 'up_datetime': up_datetime }

            if a_item['url'] is not None:
                yield scrapy.Request(response.urljoin(a_item['url']), meta={'item_1': a_item}, callback=self.parse_newscontent) #, dont_filter=True
            else:
                print("[Level-1 parse] sublink is invalid.\n")

        # if exist next page then turn to next page
        # xpath : //*[@id="ec"]/section/div[6]/div[2]/div/a[8]
        # full xpath : /html/body/section/div[6]/div[2]/div/a[8]
        level_1_next = response.xpath('//*[@class="p_next"]')              # this will be get a generator
        if level_1_next:
            # Send the next-page request, and use THIS function as callback handler
            url_next = level_1_next.xpath('@href').get()
            print('Next page ====> ', url_next)
            yield scrapy.Request(response.urljoin(url_next), callback=self.parse)
        else:
            print("[Level-1 parse] no next button.\n")


    def parse_newscontent(self, response):

        print('[LibnewsspiderSpider] Enter parse_newscontent')

        dict_item = response.meta['item_1']

        # Check the URL structure and choose the appropriate XPath
        if "news.ltn.com.tw/news" in response.url:
            content = response.xpath('//*[@id="ltnRWD"]/div[10]/section/div[4]/div[2]/p/text()').getall()
        elif "ec.ltn.com.tw/article" in response.url:
            content = response.xpath('//*[@id="talk_rwd"]/div[7]/section/div[2]/div[2]/p/text()').getall()
        else:
            print("Content not found!")
            content = []

        content = ' '.join(content).strip()
        dict_item['content'] = content
        yield dict_item
