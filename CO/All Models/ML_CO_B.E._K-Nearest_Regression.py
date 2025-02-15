#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[9]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
df


# In[10]:


df['CO_B.E'] = df['CO_B.E'] 


# In[11]:


df2=df.iloc[:,1:38]
df2


# In[12]:


print(df2.columns.tolist())


# In[13]:


X= df2
y= df['CO_B.E']


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30,30))
cor = df2.corr()
G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
plt.show()


# In[16]:


df3 = df2.drop(labels=["CO_B.E","M-Enth.vap","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
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

# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(25,25))
# cor = df4.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[17]:


X = df3
y = df['CO_B.E']


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

X_train.shape, X_test.shape


# In[19]:


#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[20]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[23]:


plt.scatter(y_test, y_predict)
plt.plot([-2,1], [-2,1],)


# In[24]:


df_pred = pd.read_csv('Cu_based_bimet_CO_Binding.csv')
df_pred


# In[25]:


x = df_pred.drop(labels=["Element","CO_B.E","M-Enth.vap","Enth.vap","M-Enth.atom","Enth.atom"], axis=1)
x


# PCA_df_Cu_pred = pd.read_csv('PCA_add_Cu_Pred.csv')
# PCA_df_Cu_pred

# combined_df_Cu_pred = pd.concat([x, PCA_df_Cu_pred], axis=1)
# df_Cu_pred = combined_df_Cu_pred
# df_Cu_pred

# In[26]:


model.predict(x)


# In[27]:


H_params = model.get_params()
H_params


# In[28]:


p = model.predict(x)
p = pd.DataFrame(p)
p


# In[29]:


output_file_CO = 'output_data_CO.xlsx'
p.to_excel(output_file_CO, index=False)
print(f"Data frame converted and saved to '{output_file_CO}'.")


# In[30]:


from scipy.stats import uniform, randint
#from sklearn.linear_model import Lasso
from scipy.stats.qmc import Sobol


# In[31]:


from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

params = {  'n_neighbors': np.arange(1, 20, 1),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': np.arange(5, 50, 5),
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan']
         }
sobol_sequence = Sobol(1)
#lasso = Lasso()
# number of times random search is run
n = 50                                              


# In[32]:


knn = KNeighborsRegressor()
average = np.array([0]*34, dtype=np.float64)
feature_importances = []

nth_run = 1
rmse_values_test = []
rmse_values_train = []
avg = 0

for i in range(n):   
    clf = RandomizedSearchCV(estimator=knn,
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

    #average += clf.best_estimator_.feature_importances_
    #feature_importances.append(clf.best_estimator_.feature_importances_)
#average = average/n
#avg = avg/n


# In[33]:


np.min(rmse_values_test), np.max(rmse_values_test)


# In[34]:


np.min(rmse_values_train), np.max(rmse_values_train)


# In[35]:


mean_rmse_train = np.mean(rmse_values_test)
print(f"Mean RMSE: {mean_rmse_train}")


# In[36]:


mean_rmse_train = np.mean(rmse_values_train)
print(f"Mean RMSE: {mean_rmse_train}")


# In[37]:


print("neg-MSE:", clf.best_score_)


# In[38]:


y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse    


# In[39]:


from sklearn.metrics import r2_score

# Add this snippet at the end of your code
y_pred_test = clf.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Best R² score for the test set: {r2}")


# In[40]:


y_predict = clf.best_estimator_.predict(X_test)


# In[41]:


y_train_predict = clf.best_estimator_.predict(X_train)
plt.figure(figsize=(12, 8))
plt.scatter(y_train, y_train_predict, facecolors='black', edgecolors='black', s=10)
plt.scatter(y_test, y_predict, marker='s', edgecolors='red',facecolors='pink' ,s=15)
plt.plot([-2,1], [-2,1],)
plt.grid(True)
plt.xlabel('DFT-calculated')
plt.ylabel('Predicted')
plt.title('(a)')

plt.show
plt.savefig('DFT vs predicted - test_size_0.15.jpg',dpi = 500)


# In[42]:


t = clf.predict(x)
t = pd.DataFrame(t)
t


# In[43]:


output_file_CO_HT = 'output_data_CO_KNN.xlsx'
t.to_excel(output_file_CO_HT, index=False)
print(f"Data frame converted and saved to '{output_file_CO_HT}'.")


# In[ ]:





# In[ ]:




