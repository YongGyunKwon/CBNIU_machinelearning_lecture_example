from scipy.stats import bernoulli 
import matplotlib.pyplot as plt 
import seaborn as sns 

data_bern=bernoulli.rvs(size=100,p=0.6)
print(data_bern)

ax=sns.distplot(data_bern,kde=False, rug=True)
ax.set(xlabel='Bernouli',ylabel='Frequency')
plt.show()