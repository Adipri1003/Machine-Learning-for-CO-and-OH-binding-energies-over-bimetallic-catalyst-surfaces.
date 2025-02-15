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


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figsize=(30,30))
##cor = df2.corr()
#G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
#plt.show()


# In[9]:


df3 = df2.drop(labels=["CO_B.E",'M-Enth.atom', 'M-Enth.vap', 'M-Sp.ht Cap', 'M-Elec.-ve', 'M-At.radii', 'M-Period', 'M-Work F.', 'At No.', 'At wt.', 'Density', 'Enth.atom', 'Enth.vap', 'Sp.ht Cap', '1st Ion E', 'Period', 'Elec.Aff'], axis=1)
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

# In[10]:


X = df3
y = df['CO_B.E']
df3 = df3.astype('float64')


# In[11]:


# Find rows with NaN or 0 values
rows_with_nan_or_zero = df3[df3.isna().any(axis=1) | (df3 == 0).any(axis=1)]

# Print the result
print(rows_with_nan_or_zero)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[13]:


#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[14]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))


# In[ ]:




