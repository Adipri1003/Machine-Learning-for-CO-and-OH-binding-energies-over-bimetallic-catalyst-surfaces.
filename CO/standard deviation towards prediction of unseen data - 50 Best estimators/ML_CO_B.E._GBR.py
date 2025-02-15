#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
df


# In[3]:


df['CO_B.E'] = df['CO_B.E'] 


# In[4]:


df2=df.iloc[:,1:38]
df2


# In[5]:


print(df2.columns.tolist())


# In[6]:


X= df2
y= df['CO_B.E']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(30,30))
#cor = df2.corr()
#G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
#plt.show()


# In[9]:


df3 = df2.drop(labels=["CO_B.E","M-Enth.fus", "M-Enth.vap","Enth.fus","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# PCA_df = pd.read_csv('PCA_add.csv')
# PCA_df

# combined_df = pd.concat([df3, PCA_df], axis=1)
# df4 = combined_df

# In[10]:


df4


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df4.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[11]:


X = df3
y = df['CO_B.E']
df3 = df3.astype('float64')


# In[12]:


# Find rows with NaN or 0 values
rows_with_nan_or_zero = df3[df3.isna().any(axis=1) | (df3 == 0).any(axis=1)]

# Print the result
print(rows_with_nan_or_zero)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[14]:


#from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[15]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# plt.scatter(y_test, y_predict)
# plt.plot([0,4], [0,4],)

# In[16]:


df_pred = pd.read_csv('Cu_based_bimet_CO_Binding.csv')
df_pred


# In[17]:


x = df_pred.drop(labels=["Element","CO_B.E","M-Enth.fus", "M-Enth.vap","Enth.fus","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
x


# PCA_df_Cu_pred = pd.read_csv('PCA_add_Cu_Pred.csv')
# PCA_df_Cu_pred

# combined_df_Cu_pred = pd.concat([x, PCA_df_Cu_pred], axis=1)
# df_Cu_pred = combined_df_Cu_pred
# df_Cu_pred

# In[18]:


model.predict(x)


# In[19]:


H_params = model.get_params()
H_params


# In[20]:


p = model.predict(x)
p = pd.DataFrame(p)
p


# In[21]:


output_file_CO = 'output_data_CO.xlsx'
p.to_excel(output_file_CO, index=False)
print(f"Data frame converted and saved to '{output_file_CO}'.")


# print(model.feature_importances_)
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='bar')
# plt.show()

# In[22]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[23]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb

params = { 'max_depth': np.arange(5,15,1),                     #13
           'learning_rate': np.arange(0.01,0.1, 0.01),      #0.325
           #'subsample': np.arange(0.95, 1.0, 0.05),             
           #'colsample_bytree': np.arange(0.99, 1, 0.01),        
           #'colsample_bylevel': np.arange(0.99, 1, 0.01),       
           'n_estimators': np.arange(100,500,25),                #450
           'alpha': uniform(0.05,0.94),                     #0.8612124738904751
           #'beta': uniform(0.05,2),                      #115.01455430429743  
         }
sobol_sequence = Sobol(1)
lasso = Lasso()
# number of times random search is run
n = 50                                              # n=20    #n_iter=50  #cv=10


# In[24]:


gbr = GradientBoostingRegressor(random_state=20)
average = np.array([0]*30, dtype=np.float64)
feature_importances = []


nth_run = 1
best_models = []
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=gbr,
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
    
  #  average += clf.best_estimator_.feature_importances_
   # feature_importances.append(clf.best_estimator_.feature_importances_)
#average = average/n
#avg = avg/n
predictions_df = pd.DataFrame()

for i, model in enumerate(best_models):
    predictions_df[f'Run_{i+1}'] = model.predict(x)

# Save predictions to an Excel file
output_file = 'output_data_50_Best_GBR.xlsx'
predictions_df.to_excel(output_file, index=False)

print(f"Predictions from 50 best RMSE models saved to '{output_file}'.")


# In[ ]:




