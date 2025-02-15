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



# In[11]:


df3 = df2.drop(labels=['M-At wt.', 'M-Density', 
                        'M-B.P', 'M-Enth.atom', 
                       'M-Enth.vap', 'M-Sp.ht Cap', 'M-cova .radii', 
                       'M-At.radii', 'M-Group', 'M-Period', 'M-Work F.', 
                       'M-Elec.Aff', 'At No.', 'At wt.', 'Density', 'M.P', 
                       'B.P', 'Enth.fus', 'Enth.atom', 'Enth.vap', 
                       'Sp.ht Cap', 'Surface.E', '1st Ion E', 
                       'cova .radii', 'Period', 
                    'Elec.Aff','OH_B.E'], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(20,20))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[12]:


PCA_df = pd.read_csv('PCA_add_7.csv')
PCA_df


# In[13]:


combined_df = pd.concat([df3, PCA_df], axis=1)
df4 = combined_df


# In[14]:


df4


# In[15]:


X = df4
y = df2['OH_B.E']


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[17]:


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[18]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[19]:


H_params = model.get_params()
H_params


# In[20]:


plt.scatter(y_test, y_predict)
plt.plot([-2,2], [-2,2],)


# In[21]:


df_pred = pd.read_csv('Cu_based_bi_OH_BE.csv')
df_pred
selected_columns = df_pred[["Element"]]
new_df = selected_columns.copy()


# In[22]:


x = df_pred.drop(labels=["Element","OH_B.E"], axis=1)
x


# model.predict(x)

# p = model.predict(x)
# p = pd.DataFrame(p)
# p

# In[25]:


print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[26]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[38]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
import xgboost as xgb

params = { 'max_depth': np.arange(8,12,1),                      #10
           'learning_rate': np.arange(0.01,0.2, 0.01),         #0.071
           'subsample': np.arange(0.7, 1.0, 0.01),              #0.75
           #'colsample_bytree': np.arange(0.9, 1, 0.02),         #0.82
           'min_child_weight': np.arange(5,15, 1),
           #'colsample_bylevel': np.arange(0.9, 1, 0.02),        #0.72
           'n_estimators': np.arange(400,1000,25),                #116
           #'reg_alpha': uniform(0.1,5),                     #0.8612124738904751
           #'reg_lambda': uniform(0.1,5),                      #115.01455430429743  
         }
sobol_sequence = Sobol(1)
lasso = Lasso()
# number of times random search is run
n = 50                                       # n=20    #n_iter=50  #cv=10


# In[39]:


xgbr = xgb.XGBRegressor(seed=20)
average = np.array([0]*17, dtype=np.float64)
feature_importances = []

nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=xgbr,
                             param_distributions=params,
                             scoring='neg_mean_squared_error',
                             n_iter=30,cv=10,random_state=np.random.RandomState(int(sobol_sequence.random(1)[0] * 2**31)),
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


# In[40]:


feature_importances 


# In[41]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[42]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[43]:


mean_rmse_test = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_test}")


# In[44]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[ ]:


avg


# In[ ]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[ ]:


average


# In[ ]:


plt.figure(figsize=(12, 8))
plt.bar(x=range(36), height=np.mean(feature_importances, axis=0), yerr=np.std(feature_importances, axis=0))
plt.xticks(ticks = range(36), labels=X_train.columns, rotation=90)


# In[ ]:


len(clf.best_estimator_.feature_importances_)
[0]*24

print("Best parameters:", clf.best_params_)



# In[ ]:


print("neg_MSE:", clf.best_score_)


# In[ ]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[ ]:


y_predict = clf.best_estimator_.predict(X_test)


# In[ ]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=10)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=15)
plt.plot([-1,2], [-1,2],)
plt.grid(True)
plt.xlabel('DFT-calculated')
plt.ylabel('Predicted')
plt.title('(d)_0.30')

plt.show
plt.savefig('DFT vs predicted_OH - test_size_0.30_xGBR.jpg',dpi = 500)


# In[ ]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8),dpi=400)
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=30)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=35)
plt.plot([-1,3], [-1,3],)
plt.grid(False)
plt.xlabel('DFT-calculated', fontsize=25)
plt.ylabel('Predicted', fontsize=25)
plt.title('(d) - Train/Test = 0.70/0.30 ', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show
plt.savefig('DFT vs predicted(xGBR) - test_size_0.30.jpg',dpi = 500)


# In[ ]:


t = clf.best_estimator_.predict(x)
t = pd.DataFrame(t)
t


# In[ ]:


output_file_OH = 'output_data_OH_xgbr_0.30.xlsx'
t.to_excel(output_file_OH, index=False)
print(f"Data frame converted and saved to '{output_file_OH}'.")


# In[ ]:


clf.best_estimator_.feature_importances_
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()
plt.savefig('Feature_Importance_best_esti_xgbr_(0.30).jpg', dpi=500)


# In[ ]:


# Assuming X is your feature matrix and clf is your trained classifier
# Replace 'dog' and 'cat' with the actual feature names you want to exclude
features_to_exclude = []

# Get feature importances
importances = clf.best_estimator_.feature_importances_

# Create a DataFrame with feature importances
feat_importances = pd.Series(importances, index=X.columns)

# Drop the specified features
feat_importances = feat_importances.drop(features_to_exclude, errors='ignore')

# Plot the feature importances
plt.figure(figsize=(12, 8))
feat_importances.nlargest(20).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

# Save the plot
plt.savefig('Feature_Importance_remaining_xgbr_(0.30).jpg', dpi=500)


# In[ ]:


avg=np.mean(feature_importances, axis=0)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(avg, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()
plt.savefig('Feature_Importance_average_xgbr_(0.30).jpg', dpi=500)


# In[ ]:


med=np.median(feature_importances, axis=0)
feat_importances = pd.Series(med, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[ ]:


std=np.std(feature_importances, axis=0)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(std, index=X.columns)
feat_importances.nlargest(15).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.show()
plt.savefig('standard deviation_xgbr_(0.30).jpg', dpi=500)


# In[ ]:





# In[ ]:




