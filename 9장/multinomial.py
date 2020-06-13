from scipy.stats import multinomial 

p=[1.0/5.0,3.0/5.0,1.0/5.0]
k=100

dist=multinomial(k,p)
cases=[20,50,30]
pr=dist.pmf(cases)

print('Case=%s Probability%.3f%%' %(cases,pr*100))