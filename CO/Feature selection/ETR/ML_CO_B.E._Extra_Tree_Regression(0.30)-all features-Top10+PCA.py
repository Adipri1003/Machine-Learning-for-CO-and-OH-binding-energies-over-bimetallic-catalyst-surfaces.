#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


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


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(30,30))
# cor = df2.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[8]:


df3 = df2.drop(labels=["CO_B.E",'M-At No.', 'M-At wt.', 'M-Density', 'M-Enth.atom', 'M-Enth.vap', 'M-Sp.ht Cap', 'M-Elec.-ve', 'M-cova .radii', 'M-At.radii', 'M-Period', 'M-Elec.Aff', 'At No.', 'At wt.', 'Density', 'B.P', 'Enth.atom', 'Enth.vap', 'Sp.ht Cap', 'Elec.-ve', 'Surface.E', '1st Ion E', 'cova .radii', 'Group', 'Period', 'Work F.', 'Elec.Aff'], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


PCA_df = pd.read_csv('PCA_add_9.csv')
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


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[27]:


#from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[28]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[29]:


plt.scatter(y_test, y_predict)
plt.plot([0,4], [0,4],)


# In[30]:


df_pred = pd.read_csv('Cu_based_bimet_CO_Binding.csv')
df_pred


# In[31]:


x = df_pred.drop(labels=["Element","CO_B.E","M-Enth.vap","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
x


# PCA_df_Cu_pred = pd.read_csv('PCA_add_Cu_Pred.csv')
# PCA_df_Cu_pred

# combined_df_Cu_pred = pd.concat([x, PCA_df_Cu_pred], axis=1)
# df_Cu_pred = combined_df_Cu_pred
# df_Cu_pred

# In[32]:


model.predict(x)


# In[33]:


H_params = model.get_params()
H_params


# In[34]:


p = model.predict(x)
p = pd.DataFrame(p)
p


# In[35]:


output_file_CO = 'output_data_CO.xlsx'
p.to_excel(output_file_CO, index=False)
print(f"Data frame converted and saved to '{output_file_CO}'.")


# In[36]:


print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[37]:


from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[38]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
#import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
params = {  'max_depth': np.arange(5, 20, 1),
            'n_estimators': np.arange(100, 400, 10),
            #'min_samples_split': np.arange(2, 20, 1),
            #'min_samples_leaf': np.arange(1, 20, 1),
            'max_features': ['auto'],                        #, 'sqrt', 'log2', None],
            'bootstrap': [True, False],
            'random_state': [42]                     
         }
sobol_sequence = Sobol(1)
lasso = Lasso()
# number of times random search is run
n = 50                                              # n=20    #n_iter=50  #cv=10


# In[39]:


etr = ExtraTreesRegressor()
average = np.array([0]*32, dtype=np.float64)
feature_importances = []

nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=etr,
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


# In[40]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[41]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[42]:


mean_rmse_test = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_test}")


# In[43]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[44]:


from sklearn.metrics import r2_score

# Add this snippet at the end of your code
y_pred_test = clf.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Best R² score for the test set: {r2}")


# In[45]:


average
    


# In[46]:


avg


# In[47]:


plt.bar(x=range(32), height=np.mean(feature_importances, axis=0), yerr=np.std(feature_importances, axis=0))
plt.xticks(ticks = range(32),labels=X_train.columns, rotation=90)


# In[48]:


len(clf.best_estimator_.feature_importances_)
[0]*24

print("Best parameters:", clf.best_params_)


# In[49]:


print("neg-MSE:", clf.best_score_)


# In[50]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[51]:


y_predict = clf.best_estimator_.predict(X_test)


# In[52]:


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
plt.savefig('DFT vs predicted(ETR) - test_size_0.29.jpg',dpi = 500)


# In[53]:


t = clf.predict(x)
t = pd.DataFrame(t)
t


# In[54]:


output_file_CO_HT = 'output_data_CO_ETR.xlsx'
t.to_excel(output_file_CO_HT, index=False)
print(f"Data frame converted and saved to '{output_file_CO_HT}'.")


# In[55]:


clf.best_estimator_.feature_importances_
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(clf.best_estimator_.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Importance - scale (0-1)')
plt.show()

plt.savefig('Feature_Importance(ETR)',dpi = 500)


# In[56]:


avg=np.mean(feature_importances, axis=0)
feat_importances = pd.Series(avg, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[57]:


med=np.median(feature_importances, axis=0)
feat_importances = pd.Series(med, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.show()


# In[58]:


std=np.std(feature_importances, axis=0)
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(std, index=X.columns)
feat_importances.nlargest(10).plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.show()


# In[ ]:





# In[ ]:




