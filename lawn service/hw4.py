import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from statsmodels.formula.api import ols

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

#1 scatterplot matrix
lawn = pd.read_csv('lawn.csv')
sns.pairplot(lawn[['lawn_ser', 'income', 'lawn_siz', 'attitude', 'teenager', 'age']])
plt.show()

#2 correlation matrix
print(lawn.corr())

#3 logistic regression
y = lawn.lawn_ser
X_train = sm.add_constant(lawn[['income', 'lawn_siz', 'attitude', 'teenager', 'age']])
est = smf.Logit(y, X_train).fit()
#est = ols("""lawn_ser~income + lawn_siz + attitude + age + teenager""", data = lawn).fit()
coef = est.summary().tables[1]
print(coef)

#5 mutiple linear regression probability
age = 45
income = 70
lawn_siz = 3
atti = 0
teen = 1
exp = np.exp(-70.4907 + 0.2868*income + 1.0647*lawn_siz - 12.744*atti - 0.2001*teen + 1.0792*age)
print(exp / (1 + exp))

#7
X_train1 = sm.add_constant(lawn[['income', 'attitude', 'teenager']])
est1 = smf.Logit(y, X_train1).fit()
coef1 = est1.summary().tables[1]
print(coef1)
exp1 = np.exp(-14.5271 + 0.16*income - 2.1904*atti + 0.2161*teen)
print(exp1 / (1 + exp1))

#8
X_train2 = sm.add_constant(lawn[['lawn_siz', 'attitude', 'teenager']])
est2 = smf.Logit(y, X_train2).fit()
coef2 = est2.summary().tables[1]
print(coef2)
exp2 = np.exp(-1.9695 + 0.5944*lawn_siz - 1.6545*atti - 0.4414*teen)
print(exp2 / (1 + exp2))

#9
X_train3 = sm.add_constant(lawn[['attitude', 'teenager', 'age']])
est3 = smf.Logit(y, X_train3).fit()
coef3 = est3.summary().tables[1]
print(coef3)
exp3 = np.exp(-11.6462 - 2.5856*atti -0.7945*teen + 0.3359*age)
print(exp3 / (1 + exp3))

#income, age, lawn_siz
X_train4 = sm.add_constant(lawn[['income', 'lawn_siz', 'age']])
est4 = smf.Logit(y, X_train4).fit()
coef4 = est4.summary().tables[1]
print(coef4)
exp4 = np.exp(-17.9927 + 0.1111*income + 0.3281*lawn_siz + 0.1213*age)
print(exp4 / (1 + exp4))

"""
The figure of scatterplot matrix indicates relationships between each two
influence factors. The relationship of each factor with willing to have lawn service
shows that, with higher income, older age, and possibly larger lawn size, the customers 
would be more willing to have the lawn service. However, the attitude and numbers of
teenagers have less influence on the decision. The forth result, which of relating attitude, teenager 
and age with lawn_ser also indicates this conclusion. About 93.55% to have the lawn service
seems not realistic. 
The first result of taking all the factors into consideration seems comprehensive, with a 
probability of 75.83% to have the service. But as mentioned above,  but attitude and teenager 
might not be neccesarry in this model.  
The second result shows a pretty low probability of about 4.2656%. As mentioned above, 
this result depends mostly on income, so it is not reasonable. But this result does indicate that income
has a larger influence when comparing with law_siz and age. 
The third result has a probability of 34.8%, also mostly depends on lawn_siz.
In the last result considered age, income and lawn_siz only shows a low probability of only 2.2469% 
to this customer comparing to other models. Since these are the factors influence the result the most,
this result is lower and more reasonable. Since this customer has a comparativly low income and lawn_siz
is small, the probability for this customer to use lawn service is low. When changing the income to 
about 100, and lawn_siz to 10, age to 60, the prbability will be improved to 97.53%! 
In conclusion, income has the largest proportion in these three main factors, while lawn_siz is ranged 2nd
and age is ranged the third. 
"""