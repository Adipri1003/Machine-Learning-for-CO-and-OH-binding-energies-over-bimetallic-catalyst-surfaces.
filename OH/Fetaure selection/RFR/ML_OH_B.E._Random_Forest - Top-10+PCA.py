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


# In[8]:


df3 = df2.drop(labels=["OH_B.E",'M-At No.', 'M-At wt.', 'M-Density', 'M-B.P', 'M-Enth.fus', 'M-Enth.vap', 'M-Sp.ht Cap', 'M-1st Ion E', 'M-cova .radii', 'M-At.radii', 'M-Period', 'M-Work F.', 'M-Elec.Aff', 'At No.', 'At wt.', 'Density', 'M.P', 'B.P', 'Enth.fus', 'Enth.atom', 'Enth.vap', 'Sp.ht Cap', 'Surface.E', '1st Ion E', 'Period', 'Elec.Aff'], axis=1)
df3


# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(20,20))
# cor = df3.corr()
# G=sns.heatmap(cor,annot=True,cmap="RdYlBu")
# plt.show()

# In[9]:


PCA_df = pd.read_csv('PCA_add_7.csv')
combined_df = pd.concat([df3, PCA_df], axis=1)
df4 = combined_df
df4


# In[10]:


X = df4
y = df2['OH_B.E']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)

X_train.shape, X_test.shape


# In[12]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
print("r2 on test data is",   r2_score(y_test, y_predict))


# In[13]:


print("RMSE:"+str(np.sqrt(mean_squared_error(y_test, y_predict))))

