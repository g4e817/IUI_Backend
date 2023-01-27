from scrapy.loader import ItemLoader
from scrapy.spiders import SitemapSpider

from crawler.items import CrawlerItem


class GutekuecheSpider(SitemapSpider):
    name = 'gutekueche'
    allowed_domains = ['gutekueche.at']
    sitemap_urls = ['https://www.gutekueche.at/sitemaps/recipe.xml.gz']

    def parse(self, response, **kwargs):
        l = ItemLoader(item=CrawlerItem(), response=response)
        l.add_value('source', response.url)
        l.add_css('categories', '.recipe-categories > a::text')
        l.add_css('title', 'h1::text')
        l.add_css('image', 'header>picture>img::attr("src")')
        return l.load_item()
