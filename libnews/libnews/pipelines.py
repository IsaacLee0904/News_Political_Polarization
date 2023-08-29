# -*- coding: utf-8 -*-
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
import datetime
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

suffix_output = "Libnews.json"

class LibnewsPipeline:
    def open_spider(self, spider):
        currentDT = datetime.datetime.now()
        dictfilename = currentDT.strftime("%Y%m%d%H%M%S_") + suffix_output
        print(dictfilename)
        self.file = open(dictfilename, 'w', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        #print(item)
        line = json.dumps(dict(item),ensure_ascii=False) + "\n"
        #in windows issue
        # self.file.write(line.encode('utf-8'))
        self.file.write(line)

        return item
