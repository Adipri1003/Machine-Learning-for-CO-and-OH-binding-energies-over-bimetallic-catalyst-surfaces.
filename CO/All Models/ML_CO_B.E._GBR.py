#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# In[24]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
df


# In[25]:


df['CO_B.E'] = df['CO_B.E'] 


# In[26]:


df2=df.iloc[:,1:38]
df2


# In[27]:


print(df2.columns.tolist())


# In[28]:


X= df2
y= df['CO_B.E']


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(30,30))
#cor = df2.corr()
#G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
#plt.show()


# In[31]:


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

# In[32]:


df4


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df4.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[33]:


X = df3
y = df['CO_B.E']
df3 = df3.astype('float64')


# In[34]:


# Find rows with NaN or 0 values
rows_with_nan_or_zero = df3[df3.isna().any(axis=1) | (df3 == 0).any(axis=1)]

# Print the result
print(rows_with_nan_or_zero)


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[36]:


#from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[37]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# plt.scatter(y_test, y_predict)
# plt.plot([0,4], [0,4],)

# In[38]:


df_pred = pd.read_csv('Cu_based_bimet_CO_Binding.csv')
df_pred


# In[39]:


x = df_pred.drop(labels=["Element","CO_B.E","M-Enth.fus", "M-Enth.vap","Enth.fus","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
x


# PCA_df_Cu_pred = pd.read_csv('PCA_add_Cu_Pred.csv')
# PCA_df_Cu_pred

# combined_df_Cu_pred = pd.concat([x, PCA_df_Cu_pred], axis=1)
# df_Cu_pred = combined_df_Cu_pred
# df_Cu_pred

# In[40]:


model.predict(x)


# In[41]:


H_params = model.get_params()
H_params


# In[42]:


p = model.predict(x)
p = pd.DataFrame(p)
p


# In[43]:


output_file_CO = 'output_data_CO.xlsx'
p.to_excel(output_file_CO, index=False)
print(f"Data frame converted and saved to '{output_file_CO}'.")


# print(model.feature_importances_)
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='bar')
# plt.show()

# In[44]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[51]:


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


# In[52]:


gbr = GradientBoostingRegressor(random_state=20)
average = np.array([0]*30, dtype=np.float64)
feature_importances = []


nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=gbr,
                             param_distributions=params,
                             scoring='neg_mean_squared_error',
                             n_iter=50,cv=10,random_state=np.random.RandomState(int(sobol_sequence.random(1)[0] * 2**31)),
                             verbose=1)
    clf.fit(X_train,y_train)
    
    print(f"Run {i + 1}: Best Estimator: {clf.best_estimator_}")
    
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
    
    average += clf.best_estimator_.feature_importances_
    feature_importances.append(clf.best_estimator_.feature_importances_)
average = average/n
avg = avg/n


# In[53]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[54]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[55]:


mean_rmse_test = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_test}")


# In[56]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[57]:


average
    


# In[58]:


avg


# In[59]:


plt.bar(x=range(30), height=np.mean(feature_importances, axis=0), yerr=np.std(feature_importances, axis=0))
plt.xticks(ticks = range(30),labels=X_train.columns, rotation=90)


# In[60]:


len(clf.best_estimator_.feature_importances_)
[0]*24

print("Best parameters:", clf.best_params_)


# In[61]:


print("neg-MSE:", clf.best_score_)


# In[62]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[63]:


from sklearn.metrics import r2_score

# Add this snippet at the end of your code
y_pred_test = clf.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Best R² score for the test set: {r2}")


# In[64]:


y_predict = clf.best_estimator_.predict(X_test)


# In[65]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=10)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=15)
plt.plot([0.5,3.5], [0.5,3.5],)
plt.grid(True)
plt.xlabel('DFT-calculated')
plt.ylabel('Predicted')
plt.title('(a)')

plt.show
plt.savefig('DFT vs predicted - test_size_0.15.jpg',dpi = 500)


# In[66]:


t = clf.predict(x)
t = pd.DataFrame(t)
t


# In[67]:


output_file_CO_HT = 'output_data_CO_GBR.xlsx'
t.to_excel(output_file_CO_HT, index=False)
print(f"Data frame converted and saved to '{output_file_CO_HT}'.")


# In[68]:


clf.best_estimator_.feature_importances_
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feat_importances.nlargest(28).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

plt.savefig('Feature_Importance',dpi = 500)


# In[69]:


avg=np.mean(feature_importances, axis=0)
feat_importances = pd.Series(avg, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[70]:


med=np.median(feature_importances, axis=0)
feat_importances = pd.Series(med, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[71]:


std=np.std(feature_importances, axis=0)
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(std, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




