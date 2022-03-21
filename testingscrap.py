import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup


def web_content_div(web_content, class_path):
    web_content_div = web_content.find_all('div', {'class': class_path})
    try:
        spans = web_content_div[0].find_all('span')
        texts = [span.get_text() for span in spans]
    except IndexError:
        texts = []
    return texts


stock_code = "TCS.NS"

url = "https://finance.yahoo.com/quote/" + stock_code + "?p=" + stock_code + "&.tsrc=fin-srch"

Stock = ['TCS.NS', 'PYPL', 'AAPL', 'AMZN', 'MSFT', 'NFLX', 'GOOG']

r = requests.get(url)
web_content = BeautifulSoup(r.text, 'lxml')

# volume = web_content.find_all('div', {'class': "D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) "
#                                                "smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY "
#                                                "smartphone_Bdc($seperatorColor)"})
#
# # volume = volume.find_all('span')
#
# print(volume)

volume = web_content.find('fin-streamer', {'data-field': "regularMarketVolume"})

print(volume)


soup = BeautifulSoup(r.text, 'lxml')
soup = soup.find(class_='D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) '
                        'smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)')
soup.find('span', attrs={'data-field': 'regularMarketVolume'})
print(soup)