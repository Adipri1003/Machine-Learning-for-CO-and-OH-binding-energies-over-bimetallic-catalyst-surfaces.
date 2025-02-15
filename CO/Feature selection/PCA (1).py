#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[2]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu.csv')
NC = df.iloc[:, 15]


# In[3]:


df2=df.iloc[:,6:9]
df2
df2 = df.drop(columns=['Element','CO_B.E','M-At No.','M-Density','M-M.P','M-Surface.E','M.P','B.P','Elec.-ve','cova .radii','At.radii','Group']) # Replace with the actual column names
df2


# In[4]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[10]:


pca = PCA(n_components=26)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[11]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[12]:


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


# In[13]:


variance_threshold = 0.95
n_components = np.argmax(explained_variance >= variance_threshold) + 1
print(f"Number of components to explain {variance_threshold * 100}% variance: {n_components}")


# In[16]:


s = pd.DataFrame(X_scaled)
s


# In[17]:


pca=PCA(n_components=8)


# In[19]:


pca.fit(X_scaled)


# In[20]:


x_pca=pca.transform(X_scaled)


# In[21]:


x_pca.shape


# In[22]:


m = pd.DataFrame(x_pca)


# In[23]:


m.columns = [f'Component_{i+1}' for i in range(8)]
m


# In[24]:


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




