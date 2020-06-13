import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self,batch_size,n_inputs,n_neurons,n_outputs):
        super(SimpleRNN,self).__init__()
        self.U=torch.randn(n_inputs,n_neurons)
        self.W=torch.randn(n_neurons,n_neurons)
        self.V=torch.randn(n_neurons,n_outputs)
        self.h=torch.randn(batch_size,n_neurons)
        self.b=torch.zeros(1,n_neurons)
        self.c=torch.zeros(1,n_outputs) 
        

    def forward(self,X):
        s=torch.mm(X,self.U)+torch.mm(self.h,self.W)+self.b 
        self.h=torch.tanh(s)
        o=torch.mm(s,self.V)+self.c 
        f=F.softmax(o,dim=0)
        return f, self.h

n_input=3
n_neurons=5
n_output=2
batch_size=4 

model=SimpleRNN(batch_size,n_input,n_neurons,n_output)

temp=[[[0,1,2],[3,4,5],[6,7,8],[9,0,1]],
      [[9,8,7],[0,0,0],[6,5,4],[3,2,1]]]

batch=torch.tensor(temp,dtype=torch.float)

for i in range(2):
    output,hidden_state = model(batch[i])
    print('output:',output)
    print('hidden state',hidden_state)

