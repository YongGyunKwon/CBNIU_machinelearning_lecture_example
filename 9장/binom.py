import numpy as np 
from scipy.stats import binom 
import matplotlib.pyplot as plt 

def run_binon(trials,n,p):
    heads=[]
    for i in range(trials):
        tosses=[np.random.random() for i in range(n)]
        heads.append(len([i for i in tosses if i<=p]))
    return heads 

successes =run_binon(10,5,0.4)
print(successes)

trials = 10000
n=20
p=0.3
x=range(0,21)
plt.plot(x,binom.pmf(x,n,p),'ro',label='Bino(10,0.3')
plt.title("binomial distribution")
plt.xlabel("no, of successes")
plt.ylabel('probability')
plt.legend()
plt.show()