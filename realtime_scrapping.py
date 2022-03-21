import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup


def web_content_div2(web_content, class_path):
    web_content_div = web_content.find('div', {'class': class_path})
    price = web_content.find('fin-streamer', {'class': "Fw(b) Fz(36px) Mb(-4px) D(ib)"}).get_text()
    # price = web_content_div.find(class_='Fw(b) Fz(36px) Mb(-4px) D(ib)').text
    spans = web_content_div.find_all('span')
    texts = [span.get_text() for span in spans]
    return price, texts


def real_time_price(stock_code):
    url = "https://finance.yahoo.com/quote/" + stock_code + "?p=" + stock_code + "&.tsrc=fin-srch"
    price, change = [], []
    try:
        r = requests.get(url)
        web_content = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36", })
        soup = BeautifulSoup(web_content.text, 'html.parser')
        texts = web_content_div2(soup, 'D(ib) Mend(20px)')
        if texts:
            price, cc = texts[0], texts[1]
            change = cc[0]+" "+cc[1]
        else:
            price, change, volume = [], []

        v = soup.find('fin-streamer', {'data-field': "regularMarketVolume"}).get_text()
        volume = str(v)

        one_year_target = soup.find('td', {'class': "Ta(end) Fw(600) Lh(14px)"}).get_text()

        pattern = soup.find('div', {'class': 'Fz(xs) Mb(4px)'})

    except ConnectionError:
        price, change, volume, one_year_target, pattern = [], [], [], [], []
    return price, change, volume, pattern, one_year_target


Stock = ['UBER', 'PYPL', 'AAPL', 'AMZN', 'MSFT', 'NFLX', 'GOOG']

print(real_time_price("NFLX"))

while True:
    info = []
    col = []
    time_stamp = datetime.datetime.now() - datetime.timedelta(hours=6)
    time_stamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
    for stock_code in Stock:
        stock_price, change, volume, latest_pattern, one_year_target = real_time_price(stock_code)
        info.append(stock_price)
        info.extend([change])
        info.extend([volume])
        info.extend([latest_pattern])
        info.extend([one_year_target])
    col = [time_stamp]
    col.extend(info)
    df = pd.DataFrame(col)
    df = df.T
    df.to_csv(str(time_stamp[0:11]) + 'stock data.csv', mode='a', header=False)
    print(col)
