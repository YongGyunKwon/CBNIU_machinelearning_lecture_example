import requests
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://finance.naver.com/item/sise_day.nhn?code=005930&page=1')
bsObject = BeautifulSoup(html, "html.parser")
quotations_links = []
closingPrices = []
quotations = []
highPrices = []
lowPrices = []
closingPricesfloat=[]; lowPricesfloat=[];highPricesfloat=[];quotationsfloat=[]

for cover in bsObject.find_all('table', {'summary':'페이지 네비게이션 리스트'}):
    for i in range(0,10,1):
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

xy = [[0]*4 for i in range(100)]

for i in range(100):
  num = int(closingPrices[i].replace(',',''))
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

xy = xy[::-1]

seq_length = 7
train_size = int(len(xy)*0.7)
train_set = np.array(xy[0:train_size])
test_set = np.array(xy[train_size - seq_length:])

print(train_set[0:5])

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)



trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX[0], ' -> ', trainY[0])



import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # 이거 패키지 깔아야함

trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x

data_dim = 4
hidden_dim = 150
output_dim = 1
learning_rate = 0.01
iterations = 5000

net = Net(data_dim, hidden_dim, output_dim, 1)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

for i in range(iterations+1):
    optimizer.zero_grad()
    outputs = net(trainX_tensor)
    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    optimizer.step()
    if i%500 == 0:
        print('iteration :', i, ' loss :', loss.item())

    net.eval()
    with torch.no_grad():
        predictY = net(testX_tensor)
        plt.plot(testY_tensor)
        plt.plot(predictY)
        plt.legend(['original', 'predict'])
        plt.show()