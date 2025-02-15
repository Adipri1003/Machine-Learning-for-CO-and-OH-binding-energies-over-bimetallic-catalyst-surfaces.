#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import time
start_time = time.time()


# In[3]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu.csv')
df


# In[4]:


df['CO_B.E'] = df['CO_B.E'] 


# In[5]:


df2=df.iloc[:,1:38]
df2


# In[6]:


print(df2.columns.tolist())


# In[7]:


X= df2
y= df['CO_B.E']


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(30,30))
# cor = df2.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

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

# df4

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(25,25))
#cor = df4.corr()
#G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
#lt.show()


# In[11]:


X = df3
y = df['CO_B.E']


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[13]:


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[14]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[15]:


plt.scatter(y_test, y_predict)
plt.plot([-2.5,0.5], [-2.5,0.5],)


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


# In[22]:


print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[23]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[24]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
import xgboost as xgb

params = { 'max_depth': np.arange(3,15,1),                     #13
           'learning_rate': np.arange(0.01,0.1, 0.01),      #0.325
           'subsample': np.arange(0.7, 1, 0.01), 
           'min_child_weight': np.arange(3,15, 1),
           #'colsample_bytree': np.arange(0.99, 1, 0.01),        
           #'colsample_bylevel': np.arange(0.99, 1, 0.01),       
           'n_estimators': np.arange(200,1000,25),                #450
           #'reg_alpha': uniform(1,3),                     #0.8612124738904751
           #'reg_lambda': uniform(1,10),                      #115.01455430429743  
         }
sobol_sequence = Sobol(1)
lasso = Lasso()
# number of times random search is run
n = 50                                             # n=20    #n_iter=50  #cv=10


# In[25]:


xgbr = xgb.XGBRegressor(seed=20)
average = np.array([0]*30, dtype=np.float64)
feature_importances = []

nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=xgbr,
                             param_distributions=params,
                             scoring='neg_mean_squared_error',
                             n_iter=100,cv=10,random_state=np.random.RandomState(int(sobol_sequence.random(1)[0] * 2**31)),
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



# In[26]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[27]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[28]:


mean_rmse_test = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_test}")


# In[29]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[30]:


average
    


# In[31]:


avg


# In[32]:


plt.bar(x=range(30), height=np.mean(feature_importances, axis=0), yerr=np.std(feature_importances, axis=0))
plt.xticks(ticks = range(30),labels=X_train.columns, rotation=90)


# In[33]:


len(clf.best_estimator_.feature_importances_)
[0]*24

print("Best parameters:", clf.best_params_)


# In[34]:


print("neg-MSE:", clf.best_score_)


# In[35]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[36]:


from sklearn.metrics import r2_score

# Add this snippet at the end of your code
y_pred_test = clf.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Best R² score for the test set: {r2}")


# In[37]:


y_predict = clf.best_estimator_.predict(X_test)


# In[38]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=10)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=15)
plt.plot([-3.0,1.5], [-3.0,1.5],)
plt.grid(True)
plt.xlabel('DFT-calculated')
plt.ylabel('Predicted')
plt.title('(d) - Train/Test = 0.70/0.30 ')

plt.show
plt.savefig('DFT vs predicted(xGBR) - test_size_0.30_mcw_md.jpg',dpi = 500)


# In[39]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8),dpi=400)
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=30)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=35)
plt.plot([-2.5,2], [-2.5,2],)
plt.grid(False)
plt.xlabel('DFT-calculated', fontsize=25)
plt.ylabel('Predicted', fontsize=25)
plt.title('(a) - Train/Test = 0.70/0.30 ', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show
plt.savefig('DFT vs predicted(xGBR) - test_size_0.30.jpg',dpi = 500)


# In[40]:


t = clf.best_estimator_.predict(x)
t = pd.DataFrame(t)
t


# In[41]:


output_file_CO_HT = 'output_data_CO_xGBR(0.30)_mcw_md.xlsx'
t.to_excel(output_file_CO_HT, index=False)
print(f"Data frame converted and saved to '{output_file_CO_HT}'.")


# In[42]:


clf.best_estimator_.feature_importances_
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

plt.savefig('Feature_Importance_xGBR_(0.30)_mcw_md.jpg',dpi = 500)


# In[43]:


# Assuming X is your feature matrix and clf is your trained classifier
# Replace 'dog' and 'cat' with the actual feature names you want to exclude
features_to_exclude = ['M-Surface.E', 'M-M.P']

# Get feature importances
importances = clf.best_estimator_.feature_importances_

# Create a DataFrame with feature importances
feat_importances = pd.Series(importances, index=X.columns)

# Drop the specified features
feat_importances = feat_importances.drop(features_to_exclude, errors='ignore')

# Plot the feature importances
plt.figure(figsize=(12, 8))
feat_importances.nlargest(29).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

# Save the plot
plt.savefig('Feature_Importance_xGBR_(0.30)_Remaining_mcw_md.jpg', dpi=500)


# In[44]:


avg=np.mean(feature_importances, axis=0)
feat_importances = pd.Series(avg, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[45]:


med=np.median(feature_importances, axis=0)
feat_importances = pd.Series(med, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[46]:


std=np.std(feature_importances, axis=0)
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(std, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.show()
plt.savefig('Standard_deviation_xGBR_(0.30)_mcw_md.jpg', dpi=500)


# In[ ]:




