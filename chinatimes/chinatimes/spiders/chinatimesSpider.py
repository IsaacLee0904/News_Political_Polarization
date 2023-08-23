# import pacakge
import scrapy
import requests
from bs4 import BeautifulSoup


class ChinatimesspiderSpider(scrapy.Spider):

    ''' basic info & setting '''
    # named the spider
    name = 'chinatimesSpider' 
    # define the domains that are allowed to be crawled
    allowed_domains = ['chinatimes.com']
    ## URL format for China Times search
    # 1. Keyword (necessary)
    # 2. Page number
    # 3. Sorted in descending order
    page_temp_url = 'https://www.chinatimes.com/search/萊豬?page=%d&chdtv'
    # range of pages
    cur_page = 1
    max_page = 178
    # start URL for the spider
    start_urls = [page_temp_url % (cur_page)]

    def parse(self, response):

        print( '[ChinatimesspiderSpider] Enter parse:', response, type(response), dir(response) )
        level_1_list = response.xpath('/html/body/div[2]/div/div[2]/div/section/div/ul/li')

        for sublink in level_1_list:

            URL = sublink.xpath('div/div/div/h3/a/@href' ).get()       # 此選擇節點下，找到的第一個 <a href=""...> tag 
            title = sublink.xpath('div/div/div/h3/a/text()' ).get()    # 此選擇節點下，找到的第一個 <a title=""...> tag
            #/html/body/div[2]/div/div[2]/div/section/div/ul/li[4]/div/div/div[2]/div/div/a
            category = sublink.xpath('div/div/div/div/div/a/text()' ).get()    # 此選擇節點下，找到的第一個 <div...> tag
            # /html/body/div[2]/div/div[2]/div/section/div/ul/li[4]/div/div/div[2]/div/time
            up_datetime = sublink.xpath('div/div/div/div/time/@datetime' ).get()  # 此選擇節點下，找到的第一個 <div...><time ...> tag

            # 目前頁面只能收集到這四個欄位
            a_item = { 'url': URL, 'title': title, 'category': category, 'up_datetime': up_datetime }

            if a_item['url'] is not None:
                yield scrapy.Request(response.urljoin(a_item['url']), meta={'item_1': a_item}, callback=self.parse_newscontent) #, dont_filter=True
            else:
                print("[Level-1 parse] sublink is invalid.\n")

        # 下一頁就是 cur page + 1
        self.cur_page += 1
        level_1_next = self.page_temp_url % (self.cur_page)

        if self.cur_page <= self.max_page:
            # Send the next-page request, and use THIS function as callback handler
            print('Next page ====> ', level_1_next)
            yield scrapy.Request(response.urljoin(level_1_next), callback=self.parse)
        else:
            print("[Level-1 parse] no next button.\n")


    def parse_newscontent(self, response):

        print( '[ChinatimesspiderSpider] Enter parse_newscontent' )

        dict_item = response.meta['item_1']  
        #level_2_root = response.xpath('//*[@id="page-top"]')
        level_2_root = response.xpath('//*[contains(@class, "article-body")]')
        level_2_content = level_2_root.xpath('p/text()')
        dict_item['content'] = ''
        for i in level_2_content:
            dict_item['content'] += i.get()

        yield  dict_item