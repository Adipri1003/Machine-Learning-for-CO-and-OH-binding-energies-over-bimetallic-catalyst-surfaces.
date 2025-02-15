#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[25]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
NC = df.iloc[:, 15]


# In[26]:


df2=df.iloc[:,6:9]
df2
df2 = df.drop(columns=['Element','CO_B.E','M-M.P', 'M-B.P', 'M-Enth.fus', 'M-Surface.E', 'M-1st Ion E', 'M-Group',
                       'M-Work F.', 'B.P', 'Enth.fus', 'At.radii','M-At.radii','M.P','M-Elec.Aff', 'M-Density', 'M-cova .radii',
                       'cova .radii','Surface.E','Work F.','Elec.-ve','M-At No.']) # Replace with the actual column names
df2


# In[29]:


print(df2.columns.tolist())


# In[30]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[31]:


pca = PCA(n_components=16)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[32]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[33]:


explained_variance = np.cumsum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
# Plotting the cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, 17), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()


# In[34]:


variance_threshold = 0.95
n_components = np.argmax(explained_variance >= variance_threshold) + 1
print(f"Number of components to explain {variance_threshold * 100}% variance: {n_components}")


# In[35]:


s = pd.DataFrame(X_scaled)
s


# In[40]:


pca=PCA(n_components=7)


# In[41]:


pca.fit(X_scaled)


# In[42]:


x_pca=pca.transform(X_scaled)


# In[43]:


x_pca.shape


# In[45]:


m = pd.DataFrame(x_pca)


# In[46]:


m.columns = [f'Component_{i+1}' for i in range(7)]
m


# In[48]:


m.to_csv('PCA_add_7.csv', index=False)


# In[49]:


n = 156
new_df = m.iloc[n:]  #Comp_1 and Comp_2 for Cu_Pred
L = pd.DataFrame(new_df)
L


# In[50]:


L.to_csv('PCA_add_Cu_Pred.csv', index=False)


# In[ ]:





# Positively correlated
# 

# In[ ]:





# In[ ]:




