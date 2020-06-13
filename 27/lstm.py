import requests
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://finance.naver.com/item/sise_day.nhn?code=005930&page=1')
bsObject = BeautifulSoup(html, "html.parser")
quotations_links=[ ] # 일별시세url 저장
closingPrices = [ ] # 종가
quotations = [ ] # 시가
highPrices = [ ] # 고가
lowPrices = [ ] # 저가
closingPricesfloat=[ ]; lowPricesfloat=[ ]; highPricesfloat=[ ]; quotationsfloat=[ ]

for cover in bsObject.find_all('table',{'summary':'페이지네비게이션리스트'}):
    for i in range(0,10,1): # 01--> 20
        quotations_links.append('https://finance.naver.com'+cover.select('a')[i].get('href'))

for index, quotations_link in enumerate(quotations_links):
    html = urlopen(quotations_link)
    bsObject = BeautifulSoup(html, "html.parser")
    for i in range(2, 70,7):
        closingPrice = bsObject.select('span')[i].text
        closingPrices.append(closingPrice)
    for i in range(4, 70,7):
        quotation = bsObject.select('span')[i].text
        quotations.append(quotation)
    for i in range(5, 70,7):
        highPrice = bsObject.select('span')[i].text
        highPrices.append(highPrice)
    for i in range(6, 70,7):
        lowPrice = bsObject.select('span')[i].text
        lowPrices.append(lowPrice)


xy = [[0]*4 for i in range(100)] # (종가저가고가시가) 저장배열

for i in range(100):
    num=int(closingPrices[i].replace(',',' '))
    closingPricesfloat.append(num)
    xy[i][0] = closingPricesfloat[i]

for i in range(100):
    num = int(lowPrices[i].replace(',',''))
    lowPricesfloat.append(num)
    xy[i][1] = lowPricesfloat[i]

for i in range(100):
    num = int(highPrices[i].replace(',',''))
    highPricesfloat.append(num)
    xy[i][2] = highPricesfloat[i]


for i in range(100):
    num = int(quotations[i].replace(',',''))
    quotationsfloat.append(num)
    xy[i][3] = quotationsfloat[i]


xy = xy[::-1] # 리스트를역순으로변환(과거--> 현재)

seq_length = 7 # 직전7일정보를이용하여주가예측
train_size = int(len(xy) * 0.7) # 70% 학습데이터
train_set = xy[0:train_size]
test_set = xy[train_size -seq_length:]

print('trainX :', trainX.shape, ' trainY : ', trainY.shape)
print(train_set[0:5])
