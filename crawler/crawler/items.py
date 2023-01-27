# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join


class CrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field(
        output_processor=TakeFirst()
    )
    categories = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=Join(',')
    )
    source = scrapy.Field(
        output_processor=TakeFirst()
    )
    image = scrapy.Field(
        output_processor=TakeFirst()
    )
