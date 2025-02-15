#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[19]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
NC = df.iloc[:, 15]


# In[20]:


df2=df.iloc[:,6:9]
df2
df2 = df.drop(columns=['Element','CO_B.E','M-M.P','M-Surface.E','M-B.P','M-Group','M-Enth.fus','Enth.fus','M.P','B.P','At.radii','M-Elec.Aff']) # Replace with the actual column names
df2


# In[21]:


print(df2.columns.tolist())


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[6]:


pca = PCA(n_components=26)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[7]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[8]:


explained_variance = np.cumsum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
# Plotting the cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, 27), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()


# In[9]:


variance_threshold = 0.95
n_components = np.argmax(explained_variance >= variance_threshold) + 1
print(f"Number of components to explain {variance_threshold * 100}% variance: {n_components}")


# In[10]:


s = pd.DataFrame(X_scaled)
s


# In[11]:


pca=PCA(n_components=8)


# In[12]:


pca.fit(X_scaled)


# In[13]:


x_pca=pca.transform(X_scaled)


# In[14]:


x_pca.shape


# In[15]:


m = pd.DataFrame(x_pca)


# In[16]:


m.columns = [f'Component_{i+1}' for i in range(8)]
m


# In[17]:


m.to_csv('PCA_add_8.csv', index=False)


# In[25]:


n = 156
new_df = m.iloc[n:]  #Comp_1 and Comp_2 for Cu_Pred
L = pd.DataFrame(new_df)
L


# In[27]:


L.to_csv('PCA_add_Cu_Pred.csv', index=False)


# In[ ]:





# Positively correlated
# 

# In[ ]:





# In[ ]:




