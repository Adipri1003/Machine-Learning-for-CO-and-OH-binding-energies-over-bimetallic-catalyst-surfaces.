#!/usr/bin/env python
# coding: utf-8

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


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30,30))
cor = df2.corr()
G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
plt.show()


# In[10]:


df3 = df2.drop(labels=["CO_B.E", 'M-At wt.','M-B.P', 'M-Enth.fus', 'M-Enth.atom', 
                       'M-Enth.vap', 'M-Sp.ht Cap', 'M-Elec.-ve','M-1st Ion E', 'M-cova .radii', 
                       'M-At.radii', 'M-Group', 'M-Period', 'M-Work F.', 'M-Elec.Aff', 'At No.', 'At wt.', 'Density',
                       'Enth.fus', 'Enth.atom', 'Enth.vap', 'Sp.ht Cap', 'Surface.E',
                       '1st Ion E','Period', 'Work F.', 'Elec.Aff', ], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[11]:


PCA_df = pd.read_csv('PCA_add_8.csv')
PCA_df


# In[12]:


combined_df = pd.concat([df3, PCA_df], axis=1)
df4 = combined_df


# In[13]:


df4


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df4.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[14]:


X = df4
y = df['CO_B.E']


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[16]:


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[17]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[18]:


plt.scatter(y_test, y_predict)
plt.plot([-3,1], [-3,1],)


# In[19]:


df_pred = pd.read_csv('Cu_based_bimet_CO_Binding.csv')
df_pred


# In[20]:


x = df_pred.drop(labels=["Element","CO_B.E",'M-At wt.','M-B.P', 'M-Enth.fus', 'M-Enth.atom', 
                       'M-Enth.vap', 'M-Sp.ht Cap', 'M-Elec.-ve','M-1st Ion E', 'M-cova .radii', 
                       'M-At.radii', 'M-Group', 'M-Period', 'M-Work F.', 'M-Elec.Aff', 'At No.', 'At wt.', 'Density',
                       'Enth.fus', 'Enth.atom', 'Enth.vap', 'Sp.ht Cap', 'Surface.E',
                       '1st Ion E','Period', 'Work F.', 'Elec.Aff'], axis=1)
x


# In[21]:


PCA_df_Cu_pred = pd.read_csv('PCA_add_Cu_Pred.csv')
PCA_df_Cu_pred


# In[22]:


combined_df_Cu_pred = pd.concat([x, PCA_df_Cu_pred], axis=1)
df_Cu_pred = combined_df_Cu_pred
df_Cu_pred


# model.predict(x)

# In[24]:


H_params = model.get_params()
H_params


# p = model.predict(x)
# p = pd.DataFrame(p)
# p

# output_file_CO = 'output_data_CO.xlsx'
# p.to_excel(output_file_CO, index=False)
# print(f"Data frame converted and saved to '{output_file_CO}'.")

# In[27]:


print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[29]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[32]:


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
n = 30                                             # n=20    #n_iter=50  #cv=10


# In[33]:


xgbr = xgb.XGBRegressor(seed=20)
average = np.array([0]*18, dtype=np.float64)
feature_importances = []

nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=xgbr,
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



# In[34]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[35]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[36]:


mean_rmse_test = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_test}")


# In[37]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[38]:


average
    


# In[39]:


avg


# In[40]:


plt.bar(x=range(18), height=np.mean(feature_importances, axis=0), yerr=np.std(feature_importances, axis=0))
plt.xticks(ticks = range(18),labels=X_train.columns, rotation=90)


# In[41]:


len(clf.best_estimator_.feature_importances_)
[0]*24

print("Best parameters:", clf.best_params_)


# In[42]:


print("neg-MSE:", clf.best_score_)


# In[43]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[44]:


y_predict = clf.best_estimator_.predict(X_test)


# In[45]:


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


# In[46]:


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


# t = clf.best_estimator_.predict(x)
# t = pd.DataFrame(t)
# t

# output_file_CO_HT = 'output_data_CO_xGBR(0.30)_mcw_md.xlsx'
# t.to_excel(output_file_CO_HT, index=False)
# print(f"Data frame converted and saved to '{output_file_CO_HT}'.")

# In[47]:


clf.best_estimator_.feature_importances_
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feat_importances.nlargest(18).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

plt.savefig('Feature_Importance_xGBR_(0.30)_mcw_md.jpg',dpi = 500)


# In[48]:


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
feat_importances.nlargest(16).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

# Save the plot
plt.savefig('Feature_Importance_xGBR_(0.30)_new_PCA_8.jpg', dpi=500)


# In[49]:


avg=np.mean(feature_importances, axis=0)
feat_importances = pd.Series(avg, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[50]:


med=np.median(feature_importances, axis=0)
feat_importances = pd.Series(med, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[51]:


std=np.std(feature_importances, axis=0)
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(std, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.show()
plt.savefig('Standard_deviation_xGBR_(0.30)_mcw_md.jpg', dpi=500)


# In[ ]:


end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")


# In[ ]:




