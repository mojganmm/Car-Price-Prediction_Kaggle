
# coding: utf-8

# In[1]:


#car feature data collected using "Edmunds car API".11,915 samples of used & new cars.New cars average MSRP, old cars TMV value 

#import all required libraries

import os
import pandas as pd
import numpy as np

#visulization 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib

#data preprocessing

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import norm
from scipy import stats
from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from mlinsights.mlmodel import QuantileLinearRegression

import xgboost
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score


# In[2]:


#read data
os.chdir("C:\\Users\\38812\\Desktop\\kaggle\\car_price")

def ReadFromCSV(FilePath, sckipLines = 0, sep = ','):
#   Read and clean the csv file in FilePath ignoring the first sckipLines lines. (omits non-ascii characters from columns' name)
	xa = pd.read_csv(FilePath, header=sckipLines, sep = sep, low_memory=False)
	#xa.columns = [CleanStr(x) for x in list(xa.columns)]
	return xa


# In[3]:


#Exploratory Data Analysis

df_tr=ReadFromCSV('train.csv', sckipLines = 0, sep = ',') 
#print (df_tr.shape)# 11,914 rows & 16 cols

df_tr['MSRP'].describe()  #range ($2k,$2m)
ax = sns.distplot(df_tr['MSRP'])
#print (df_tr.shape)


# In[4]:


# Preprocessing Data
all_data=df_tr.copy()
total =all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[5]:


#dealing with missing data by dropping cols with more than 45% missing values

all_data= all_data.drop((missing_data[missing_data['Percent'] > 0.5]).index,1) 

all_data_noncons=all_data.loc[:,all_data.apply(pd.Series.nunique) != 1] #remove cons vars
print (all_data.shape)


# In[6]:


#all_data['Market Category'] = all_data['Market Category'].fillna('NoCategory')

all_data.dropna(subset=['Market Category'], how='any', inplace = True)
all_data['Engine HP'] = all_data['Engine HP'].fillna(all_data['Engine HP'].mode()[0])
all_data['Engine Cylinders'] = all_data['Engine Cylinders'].fillna(all_data['Engine Cylinders'].mode()[0])
all_data['Number of Doors'] = all_data['Number of Doors'].fillna(all_data['Number of Doors'].mode()[0])
all_data['Engine Fuel Type'] = all_data['Engine Fuel Type'].fillna(all_data['Engine Fuel Type'].mode()[0])

nacols=all_data.columns[all_data.isna().any()].tolist()
print (len(nacols))
print (all_data.shape[0])

all_data['Year'] = all_data['Year'].astype(str) #convert numeric to category #1990 to 2017
all_data['Number of Doors'] = all_data['Number of Doors'].astype(str) #convert numeric to category


# In[7]:


#Correlation matrix (heatmap style)

corr1 = all_data[all_data.columns.values].corr()
f, ax = plt.subplots(figsize=(8,7))
sns.heatmap(corr1, annot=True)


# In[8]:


ax = sns.pairplot(all_data[['Engine HP','Engine Cylinders', 'MSRP']])


# In[9]:


##bivariate analysis 
var = 'Engine HP'
data = pd.concat([all_data['MSRP'], all_data[var]], axis=1)
data.plot.scatter(x=var, y='MSRP', ylim=(0,800000));
p1=all_data.sort_values(by = 'Engine HP', ascending = False)[:5] #sort descending and find first 2 max of
print (p1['Engine HP'], 'p1')
#bivariate analysis saleprice/TotalBsmtSF
var = 'Engine Cylinders'
data = pd.concat([all_data['MSRP'], all_data[var]], axis=1)
data.plot.scatter(x=var, y='MSRP', ylim=(0,800000));
p2=all_data.sort_values(by = 'Engine Cylinders', ascending = False)[:5] #sort descending and find  max of 
print (p2['Engine Cylinders'], 'p2')


# In[10]:


#deleting outlier points
var = 'Engine HP'
all_data = all_data.drop(all_data[all_data['Engine HP'] == 11362].index) 
all_data = all_data.drop(all_data[all_data['Engine HP'] == 11363 ].index)
all_data = all_data.drop(all_data[all_data['Engine HP'] == 11364 ].index)
all_data = all_data.drop(all_data[all_data['Engine HP'] == 1630 ].index)
all_data = all_data.drop(all_data[all_data['Engine HP'] == 1629 ].index)

all_data = all_data.drop(all_data[all_data['Engine Cylinders'] ==7556 ].index)
all_data = all_data.drop(all_data[all_data['Engine Cylinders'] == 11100 ].index)

data = pd.concat([all_data['MSRP'], all_data[var]], axis=1)
data.plot.scatter(x=var, y='MSRP', ylim=(0,700000));


# In[ ]:


#all_data=all_data.drop('Popularity', axis=1) ## of appearnace of car brand in tweets within specific time frame


# In[11]:


## Data Transformation using log scale

#features=all_data.drop('MSRP', axis=1).copy()

features=all_data.copy()

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.75]
skew_index = high_skew.index

features[skew_index] = np.log1p(features[skew_index])


# In[ ]:


## Data Transformation using Box_Cox scale

features=all_data.copy()

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.75]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1)) #boxcox_normmax: Compute optimal Box-Cox transform parameter for input data.


# In[12]:


features= pd.get_dummies(features).reset_index(drop=True)  # Getting Dummies from all other categorical vars
print (features.shape)


# In[13]:


#creating matrices for sklearn:

features = features.reset_index(level=0, drop=True).reset_index()
all_data = all_data.reset_index(level=0, drop=True).reset_index()
X_train = features.drop('MSRP', axis=1).copy()

#y = np.log(all_data['MSRP'])
y=features['MSRP']

#print (y.type)

X_tr, X_te, Y_tr, Y_te = train_test_split(X_train, y, test_size=0.25, random_state=42)

#print (X_tr.shape)
#print (Y_tr.shape)


# In[14]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_tr, Y_tr, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_tr, Y_tr)
rmse_cv(model_lasso).mean()

print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tr, model_lasso.predict(X_tr)))  
print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_te, model_lasso.predict(X_te)))  
print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tr, model_lasso.predict(X_tr))))
print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_te, model_lasso.predict(X_te))))

R2_tr = r2_score(Y_tr, model_lasso.predict(X_tr))
print (R2_tr)

R2_te = r2_score(Y_te, model_lasso.predict(X_te))
print (R2_te)


# In[15]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_tr), "true":Y_tr})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
df_results = pd.DataFrame({'Predicted MSRP': model_lasso.predict(X_te), 'Actual MSRP': Y_te})
df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


# coef = pd.Series(model_lasso.coef_, index = X_tr.columns)
# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# imp_coef = pd.concat([coef.sort_values().head(10),
#                      coef.sort_values().tail(10)])
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")

ax = sns.distplot(y)


# In[ ]:


#if log transformation used:

features_b=features[features['MSRP']<9]
#features_b['Market Category'].mode()

features_e=features[features['MSRP']>9]


# In[ ]:


# #if boxcox transformation used:

# features_b=features[features['MSRP']<25]
# #features_b['Market Category'].mode()

# features_e=features[features['MSRP']>25]


# In[ ]:


#creating matrices for sklearn:

features_b = features_b.reset_index(level=0, drop=True).reset_index()

features_e= features_e.reset_index(level=0, drop=True).reset_index()

X_train_b = features_b.drop('MSRP', axis=1).copy()
y_b=features_b['MSRP']

X_train_e = features_e.drop('MSRP', axis=1).copy()
y_e=features_e['MSRP']
#print (y.type)


# In[ ]:


ax = sns.distplot(y_b)
ax = sns.distplot(y_e)


# In[ ]:


X_trb, X_teb, Y_trb, Y_teb = train_test_split(X_train_b, y_b, test_size=0.25, random_state=42)

print (X_trb.shape)
print (Y_trb.shape)

X_tre, X_tee, Y_tre, Y_tee = train_test_split(X_train_e, y_e, test_size=0.25, random_state=42)

print (X_tre.shape)
print (Y_tre.shape)


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_trb, Y_trb, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_trb, Y_trb)
rmse_cv(model_lasso).mean()

print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_trb, model_lasso.predict(X_trb)))  
print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_teb, model_lasso.predict(X_teb)))  
print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_trb, model_lasso.predict(X_trb))))
print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_teb, model_lasso.predict(X_teb))))

R2_tr = r2_score(Y_trb, model_lasso.predict(X_trb))
print (R2_tr)

R2_te = r2_score(Y_teb, model_lasso.predict(X_teb))
print (R2_te)

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_trb), "true":Y_trb})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")

preds["Standardize residuals"] = (preds["true"] - preds["preds"])/(np.sqrt(metrics.mean_squared_error(Y_trb, model_lasso.predict(X_trb))))
preds.plot(x = "preds", y = "Standardize residuals",kind = "scatter")

df_results = pd.DataFrame({'Predicted MSRP': model_lasso.predict(X_teb), 'Actual MSRP': Y_teb})
df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_tre, Y_tre)
rmse_cv(model_lasso).mean()

print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tre, model_lasso.predict(X_tre)))  
print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_tee, model_lasso.predict(X_tee)))  
print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tre, model_lasso.predict(X_tre))))
print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tee, model_lasso.predict(X_tee))))

R2_tr = r2_score(Y_tre, model_lasso.predict(X_tre))
print (R2_tr)

R2_te = r2_score(Y_tee, model_lasso.predict(X_tee))
print (R2_te)

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_tre), "true":Y_tre})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
preds["Standardize residuals"] = (preds["true"] - preds["preds"])/(np.sqrt(metrics.mean_squared_error(Y_tre, model_lasso.predict(X_tre))))
preds.plot(x = "preds", y = "Standardize residuals",kind = "scatter")
df_results = pd.DataFrame({'Predicted MSRP': model_lasso.predict(X_tee), 'Actual MSRP': Y_tee})
df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


#from mlinsights.mlmodel import PiecewiseRegressor
#from sklearn.tree import DecisionTreeRegressor


clqs = {}
for qu in [0.25, 0.5, 0.85]:
    clq = QuantileLinearRegression(quantile=qu)
    clq.fit(X_tr, Y_tr)
    clqs['q=%1.2f' % qu] = clq
    print (clq)
    print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tr, clq.predict(X_tr)))  
    print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_te, clq.predict(X_te)))  
    print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tr, clq.predict(X_tr))))
    print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_te, clq.predict(X_te))))

    R2_tr = r2_score(Y_tr, clq.predict(X_tr))
    print (R2_tr)

    R2 = r2_score(Y_te, clq.predict(X_te))
    print (R2)
    
    #let's look at the residuals as well:
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

    preds = pd.DataFrame({"preds":clq.predict(X_tr), "true":Y_tr})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    preds["Standardize residuals"] = (preds["true"] - preds["preds"])/(np.sqrt(metrics.mean_squared_error(Y_tr, clq.predict(X_tr))))
    preds.plot(x = "preds", y = "Standardize residuals",kind = "scatter")
    df_results = pd.DataFrame({'Predicted MSRP': clq.predict(X_tr), 'Actual MSRP': Y_tr})
    df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


# clq = QuantileLinearRegression()
# clq.fit(X_trb, Y_trb)

# print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_trb, clq.predict(X_trb)))  
# print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_teb, clq.predict(X_teb)))  
# print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_trb, clq.predict(X_trb))))
# print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_teb, clq.predict(X_teb))))

# R2_tr = r2_score(Y_trb, clq.predict(X_trb))
# print (R2_tr)

# R2_te = r2_score(Y_teb, clq.predict(X_teb))
# print (R2_te)
# #let's look at the residuals as well:
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# preds = pd.DataFrame({"preds":clq.predict(X_trb), "true":Y_trb})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")

# df_results = pd.DataFrame({'Predicted MSRP': clq.predict(X_trb), 'Actual MSRP': Y_trb})
# df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')

# df_results_te = pd.DataFrame({'Predicted MSRP': clq.predict(X_teb), 'Actual MSRP': Y_teb})
# df_results_te.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


# clqe = QuantileLinearRegression()
# clqe.fit(X_tre, Y_tre)

# print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tre, clqe.predict(X_tre)))  
# print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_tee, clqe.predict(X_tee)))  
# print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tre, clqe.predict(X_tre))))
# print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tee, clqe.predict(X_tee))))

# R2_tr = r2_score(Y_tre, clqe.predict(X_tre))
# print (R2_tr)

# R2 = r2_score(Y_tee, clqe.predict(X_tee))
# print (R2)

# #let's look at the residuals as well:
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# preds = pd.DataFrame({"preds":clqe.predict(X_tre), "true":Y_tre})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")

# df_results = pd.DataFrame({'Predicted MSRP': clqe.predict(X_tre), 'Actual MSRP': Y_tre})
# df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


# target=features.as_matrix(columns=['MSRP'])

# from sklearn.mixture import GaussianMixture      # 1. Choose the model class
# model = GaussianMixture(2,covariance_type='full',random_state=0)  # 2. Instantiate the model with hyperparameters
# model.fit(target)                    # 3. Fit to data. Notice y is not specified!
# y_gmm = model.predict(target)        # 4. Determine cluster labels

# features['cluster'] = y_gmm
# ax = sns.distplot(y_gmm)


# In[ ]:


# clqs = {}
# for qu in [0.25, 0.5, 0.75, 0.9]:
#     clq = QuantileLinearRegression(quantile=qu)
#     clq.fit(X_tr, Y_tr)
#     clqs['q=%1.2f' % qu] = clq
#     print (clq)
#     print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tr, clq.predict(X_tr)))  
#     print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_te, clq.predict(X_te)))  
#     print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tr, clq.predict(X_tr))))
#     print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_te, clq.predict(X_te))))

#     R2_tr = r2_score(Y_tr, clq.predict(X_tr))
#     print (R2_tr)

#     R2 = r2_score(Y_te, clq.predict(X_te))
#     print (R2)
    
#     #let's look at the residuals as well:
#     matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

#     preds = pd.DataFrame({"preds":clq.predict(X_tr), "true":Y_tr})
#     preds["residuals"] = preds["true"] - preds["preds"]
#     preds.plot(x = "preds", y = "residuals",kind = "scatter")
#     df_results = pd.DataFrame({'Predicted MSRP': clq.predict(X_tr), 'Actual MSRP': Y_tr})
#     df_results.plot('Actual MSRP', 'Predicted MSRP', kind='scatter')


# In[ ]:


# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, X_tr, Y_tr, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)
# model_ridge = Ridge() 

# alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
# cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()for alpha in alphas]
# cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation - Just Do It")
# plt.xlabel("alpha")
# plt.ylabel("rmse")

# cv_ridge.min() #alpha=0.123


# model_ridge = Ridge(alpha=0.123).fit(X_tr, Y_tr) 


# print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tr, model_ridge.predict(X_tr)))  
# print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_te, model_ridge.predict(X_te)))  
# print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tr, model_ridge.predict(X_tr))))
# print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_te, model_ridge.predict(X_te))))

# acc = r2_score(Y_te, model_ridge.predict(X_te))
# print (acc)


# In[ ]:


# #let's look at the residuals as well:
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# preds = pd.DataFrame({"preds":model_ridge.predict(X_tr), "true":Y_tr})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")


# In[ ]:


# # train the model
# RFregbbk = RandomForestRegressor(n_estimators=1000, oob_score=True, criterion='mse', max_depth=6, min_samples_split=10,
#                                  min_samples_leaf=6, max_features='auto', max_leaf_nodes=None, bootstrap=True,
#                                  n_jobs=-1, random_state=None, verbose=0, warm_start=False).fit(X_tr, Y_tr)  # n_estimators=number of trees, the higher the better perfomance lower speed

# print ('Training Mean Absolute Error:', metrics.mean_absolute_error(Y_tr, RFregbbk.predict(X_tr)))  
# print ('Testing Mean Squared Error:', metrics.mean_squared_error(Y_te, RFregbbk.predict(X_te)))  
# print ('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_tr, RFregbbk.predict(X_tr))))
# print ('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_te, RFregbbk.predict(X_te))))
      
# # Feature_Importances
# names = X_tr.columns

# R2_tr = r2_score(Y_tr, RFregbbk.predict(X_tr))
# print (R2_tr)
# R2 = r2_score(Y_te, RFregbbk.predict(X_te))
# print (R2)


# In[ ]:


# #let's look at the residuals as well:
# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# preds = pd.DataFrame({"preds":RFregbbk.predict(X_tr), "true":Y_tr})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")
# feature_importances = pd.DataFrame(RFregbbk.feature_importances_,
#                                    index = X_tr.columns,
#                                     columns=['importance']).sort_values('importance', ascending=False)
# print (feature_importances[0:10])
# df_results = pd.DataFrame({'Predicted Price': RFregbbk.predict(X_tr), 'Actual Price': Y_tr})
# df_results.plot('Actual Price', 'Predicted Price', kind='scatter')

