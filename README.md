# -
给产品做市调，于是第一次爬虫就开始啦
from urllib.request import Request
from urllib.request import urlopen
from bs4 import BeautifulSoup

UA = [
   "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36",
   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1",
   "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0",
   "Opera/9.80 (Windows NT 6.1; U; zh-cn) Presto/2.9.168 Version/11.50"
      ]

headers = {"User-Agent": random.choice(UA),
           "authority": "www.amazon.com",
           "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
           "accept-language": "zh-CN,zh;q=0.9",
           "referer": "https://www.amazon.com/s?k=vegetable+chopper&ref=nb_sb_noss_1"          
           }
