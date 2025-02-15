#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv('Dataset_A3B_OH_343.csv')
df


# In[3]:


df2=df.iloc[0:69,1:38]
df2


# In[4]:


print(df2.columns.tolist())


# In[5]:


X= df2
y= df2['OH_B.E']


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(25,25))
cor = df2.corr()
G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
plt.show()


# In[8]:


df3 = df2.drop(labels=["OH_B.E"], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(20,20))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[9]:


X = df3
y = df2['OH_B.E']


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[11]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[12]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[13]:


H_params = model.get_params()
H_params


# In[14]:


plt.scatter(y_test, y_predict)
plt.plot([-2,2], [-2,2],)


# In[15]:


df_pred = pd.read_csv('Cu_based_bi_OH_BE.csv')
df_pred
selected_columns = df_pred[["Element"]]
new_df = selected_columns.copy()


# In[16]:


x = df_pred.drop(labels=["Element","OH_B.E"], axis=1)
x


# In[17]:


model.predict(x)


# In[18]:


p = model.predict(x)
p = pd.DataFrame(p)
p


# In[19]:


print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[20]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[21]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor

params = {  'max_depth': np.arange(3, 20, 1),
            'n_estimators': np.arange(50, 200, 10),
            #'min_samples_split': np.arange(2, 20, 1),
            #'min_samples_leaf': np.arange(1, 20, 1),
            #'max_features': ['auto', 'sqrt', 'log2', None],
            #'bootstrap': [True, False],
            'random_state': [42]                     
         }
sobol_sequence = Sobol(1)
lasso = Lasso()
# number of times random search is run
n = 50                                      


# In[22]:


etr = ExtraTreesRegressor()
average = np.array([0]*32, dtype=np.float64)
feature_importances = []

nth_run = 1
best_models = []
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=etr,
                             param_distributions=params,
                             scoring='neg_mean_squared_error',
                             n_iter=30,cv=10,random_state=np.random.RandomState(int(sobol_sequence.random(1)[0] * 2**31)),
                             verbose=1)
    clf.fit(X_train,y_train)
    
    print(f"Run {i + 1}: Best Estimator: {clf.best_estimator_}")
    best_models.append(clf.best_estimator_)
    
    if (i + 1) % nth_run == 0:
        # Predictions on test set
        y_pred_test = clf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        rmse_values_test.append(rmse_test)
        
        # Predictions on training set
        y_pred_train = clf.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        rmse_values_train.append(rmse_train)

        print(f"RMSE for every {nth_run}th run - Training: {rmse_train}, Test: {rmse_test}")
        nth_run += 1
    
    
   # average += clf.best_estimator_.feature_importances_
   # feature_importances.append(clf.best_estimator_.feature_importances_)
#average = average/n
#avg = avg/n
predictions_df = pd.DataFrame()

for i, model in enumerate(best_models):
    predictions_df[f'Run_{i+1}'] = model.predict(x)

# Save predictions to an Excel file
output_file = 'output_data_50_Best_ETR.xlsx'
predictions_df.to_excel(output_file, index=False)

print(f"Predictions from 50 best RMSE models saved to '{output_file}'.")


# In[ ]:




