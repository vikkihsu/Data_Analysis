import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import decimal

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors

import statsmodels.api as sm
import statsmodels.formula.api as smf

def median(lst):
    new_lst = sorted(lst)
    if len(lst) != 1:
        if len(new_lst) % 2 != 0:
            return new_lst[int((len(new_lst)+1)/2-1)]
        else:
            return (new_lst[len(new_lst)/2-1]+new_lst[len(new_lst)/2])/2.0
    else:
        return lst[0]

def factorize(lst):
	list_01 = []
	for n in range(len(lst)):
		if lst[n] >= median(lst): list_01.append(1)
		else: list_01.append(0)
	return list_01

#factorizing mpg to 0 or 1
auto = pd.read_csv('Auto.csv')
auto['mpg01'] = factorize(list(auto.mpg))
#print(auto['mpg01'])

#scatter mpg with cylinders, displacement, horsepower, weight and acceleration
fig,(ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
# Take a fraction of the samples where target value (default) is 'no'
auto_no = auto[auto.mpg01 == 0].sample(frac=0.15)
# Take all samples  where target value is 'yes'
auto_yes = auto[auto.mpg01 == 1]
auto_ = auto_no.append(auto_yes)

ax1.scatter(auto_[auto_.mpg >= median(list(auto.mpg))].mpg, 
			auto_[auto_.mpg >= median(list(auto.mpg))].cylinders,
			s=40, c='orange', marker='+',
            linewidths=1)
ax1.scatter(auto_[auto_.mpg < median(list(auto.mpg))].mpg,
			auto_[auto_.mpg < median(list(auto.mpg))].cylinders, 
			s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='none')
ax1.set_ylim(ymin=0)
ax1.set_ylabel('mpg')
ax1.set_xlim(xmin=0)
ax1.set_xlabel('cylinders')

ax2.scatter(auto_[auto_.mpg >= median(list(auto.mpg))].mpg, 
			auto_[auto_.mpg >= median(list(auto.mpg))].displacement,
			s=40, c='orange', marker='+',
            linewidths=1)
ax2.scatter(auto_[auto_.mpg < median(list(auto.mpg))].mpg,
			auto_[auto_.mpg < median(list(auto.mpg))].displacement, 
			s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='none')
ax2.set_ylim(ymin=0)
ax2.set_ylabel(' ')
ax2.set_xlim(xmin=0)
ax2.set_xlabel('displacement')

ax3.scatter(auto_[auto_.mpg >= median(list(auto.mpg))].mpg, 
			auto_[auto_.mpg >= median(list(auto.mpg))].horsepower,
			s=40, c='orange', marker='+',
            linewidths=1)
ax3.scatter(auto_[auto_.mpg < median(list(auto.mpg))].mpg,
			auto_[auto_.mpg < median(list(auto.mpg))].horsepower, 
			s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='none')
ax3.set_ylim(ymin=0)
ax3.set_ylabel(' ')
ax3.set_xlim(xmin=0)
ax3.set_xlabel('horsepower')

ax4.scatter(auto_[auto_.mpg >= median(list(auto.mpg))].mpg, 
			auto_[auto_.mpg >= median(list(auto.mpg))].weight,
			s=40, c='orange', marker='+',
            linewidths=1)
ax4.scatter(auto_[auto_.mpg < median(list(auto.mpg))].mpg,
			auto_[auto_.mpg < median(list(auto.mpg))].weight, 
			s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='none')
ax4.set_ylim(ymin=0)
ax4.set_ylabel(' ')
ax4.set_xlim(xmin=0)
ax4.set_xlabel('weight')

ax5.scatter(auto_[auto_.mpg >= median(list(auto.mpg))].mpg, 
			auto_[auto_.mpg >= median(list(auto.mpg))].acceleration,
			s=40, c='orange', marker='+',
            linewidths=1)
ax5.scatter(auto_[auto_.mpg < median(list(auto.mpg))].mpg,
			auto_[auto_.mpg < median(list(auto.mpg))].acceleration, 
			s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='none')
ax5.set_ylim(ymin=0)
ax5.set_ylabel(' ')
ax5.set_xlim(xmin=0)
ax5.set_xlabel('acceleration')

#plt.show()

#training set and test set
#mpg with acceleration
#scikit-learn
X_train = auto.acceleration.reshape(-1,1) 
y = auto.mpg01
X_test = np.arange(auto.acceleration.min(), auto.acceleration.max()).reshape(-1,1)
classify = skl_lm.LogisticRegression(solver='newton-cg')
classify.fit(X_train,y)
prob = classify.predict_proba(X_test)

fig, (ax1, ax2) = plt.subplots(1,2)
sns.regplot(auto.acceleration, auto.mpg01, order=1, ci=None,
            scatter_kws={'color':'orange'},
            line_kws={'color':'lightblue', 'lw':2}, ax=ax1)
ax2.scatter(X_train, y, color='orange')
ax2.plot(X_test, prob[:,1], color='lightblue')

for ax in fig.axes:
    ax.hlines(1, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.hlines(0, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.set_ylabel('Probability of mpg')
    ax.set_xlabel('acceleration')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax.set_xlim(xmin=5)
#plt.show()
print(classify)
print('classes: ',classify.classes_)
print('coefficients: ',classify.coef_)
print('intercept :', classify.intercept_)

#statsmodel
X_train = sm.add_constant(auto.acceleration)
estimate = smf.Logit(y.ravel(), X_train).fit()
print(estimate.summary().tables[1])

#mutiple
X_train = sm.add_constant(auto[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']])
estimate = smf.Logit(y, X_train).fit()
print(estimate.summary().tables[1])

#LDA
X = auto[['cylinders','acceleration']]
y = auto.mpg01

lda = LinearDiscriminantAnalysis(solver = 'svd')
y_pred = lda.fit(X, y).predict(X)

auto_ = pd.DataFrame({'True status' : y, 
						'Predict status' : y_pred})
auto_.replace(to_replace = {0 : 'No', 1: 'Yes'}, inplace = True)
print(auto_.groupby(['Predict status', 'True status']).size().unstack('True status'))

print(lda.priors_)
print(lda.means_)
print(lda.coef_)

#QDA
qda = QuadraticDiscriminantAnalysis()
pred = qda.fit(X, y).predict(X)
print(qda.priors_)
print(qda.means_)

#logistic regression
regr = skl_lm.LogisticRegression()
pred = regr.fit(X, y).predict(X)
pred_p = regr.predict_proba(X)

auto_ = pd.DataFrame({'True status' : y, 
						'Predict status' : pred})
auto_.replace(to_replace = {0 : 'No', 1: 'Yes'}, inplace = True)
print(auto_.groupby(['Predict status', 'True status']).size().unstack('True status'))


#KNN
knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
pred = knn.fit(X, y).predict(X)
