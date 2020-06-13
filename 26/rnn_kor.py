#-*-coding: utf-8 -*-

import numpy as np 

data =open('26/korean_bible.txt','rt',encoding='UTF-8').read() 
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
print('Total number of chars : %d, Distinguished chars :%d' %(data_size,vocab_size))

char_to_ix={ch:i for i ,ch in enumerate(chars)}
ix_to_char={i:ch for i ,ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)

hidden_size=100
seq_length=25
learing_rate=1e-1


U=np.random.randn(hidden_size,vocab_size)
W=np.random.randn(hidden_size,hidden_size)
V=np.random.randn(vocab_size,hidden_size)
b=np.zeros((hidden_size,1))
c=np.zeros((vocab_size,1))

def evalLossGradientHstate(inputs,targets,hprev):
    xs,hs,os,ys={ },{ },{ },{ }
    hs[-1]=np.copy(hprev)
    loss=0

    for t in range(len(inputs)):
        xs[t]=np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(np.dot(U,xs[t])+np.dot(W,hs[t-1])+b)
        os[t]=np.dot(V,hs[t])+c
        ys[t]=np.exp(os[t])/np.sum(np.exp(os[t]))
        loss+=-np.log(ys[t][targets[t],0])

    dU,dW,dV=np.zeros_like(U),np.zeros_like(W),np.zeros_like(V)
    db,dc=np.zeros_like(b),np.zeros_like(c)
    dhh=np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        do=np.copy(ys[t])
        do[targets[t]]-=1
        dV+=np.dot(do,hs[t].T)
        dc+=do
        dh=np.dot(V.T,do)+dhh
        dhHs=(1-hs[t]*hs[t])*dh
        db+=dhHs
        dU+=np.dot(dhHs,xs[t].T)
        dW+=np.dot(dhHs,hs[t-1].T)
        dhh=np.dot(W.T,dhHs)

    for dparam in [dU,dW,dV,db,dc]:
        np.clip(dparam,-5,5,out=dparam)

    return loss,dU,dW,dV,db,dc,hs[len(inputs)-1]

def generateText(h,seed_ix,n):
    x=np.zeros((vocab_size,1))
    x[seed_ix]=1

    ixes=[ ]
    
    for t in range(n):
        h=np.tanh(np.dot(U,x)+np.dot(W,h)+b)
        o=np.dot(V,h)+c
        y=np.exp(o)/np.sum(np.exp(o))
        ix=np.random.choice(range(vocab_size),p=y.ravel())
        x=np.zeros((vocab_size,1))
        x[ix]=1
        ixes.append(ix)

        txt=''.join(ix_to_char[ix] for ix in ixes)
        print('---\n %s \n ---' %(txt, ))

print('Generate chars before training')
hprev=np.zeros((hidden_size,1))
generateText(hprev,0,200)

n,p=0,0 
mU,mW,mV=np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
mb,mc=np.zeros_like(b),np.zeros_like(c)
smooth_loss=-np.log(1.0/vocab_size)*seq_length

while n<=1000* 1500:
    if p+seq_length+1>=len(data) or n ==0:
        hprev=np.zeros((hidden_size,1))
        p=0
    inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    loss,dU,dW,dV,db,dc,hprev = evalLossGradientHstate(inputs,targets,hprev)
    smooth_loss=smooth_loss*0.999+loss*0.001

    if n%5000==0:
        print('iter %d, loss %f' %(n,smooth_loss))
        generateText(hprev,inputs[0],200)

    for param, dparam, mem in zip([U,W,V,b,c],[dU,dW,dV,db,dc],[mU,mW,mV,mb,mc]):
        mem+=dparam*dparam
        param+=-learing_rate*dparam/np.sqrt(mem+1e-8)
    
    p+=seq_length
    n+=1