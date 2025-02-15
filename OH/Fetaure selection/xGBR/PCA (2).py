#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[59]:


df= pd.read_csv('Dataset_A3B_OH_343.csv')


# In[60]:


df2=df.iloc[0:69,1:38]
df2 = df2.drop(columns=['M-At No.', 'M-M.P', 'M-Enth.fus', 'M-Elec.-ve', 'M-Surface.E', 
 'M-1st Ion E', 'M-Work F.', 'M-Elec.Aff', 'Density', 'M.P', 
 'Enth.fus', 'Enth.atom', 'Elec.-ve', 'Surface.E', 
 '1st Ion E', 'cova .radii', 'At.radii', 'Group', 'Work F.', 
 'Elec.Aff', 'OH_B.E']) 
df2


# In[42]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[43]:


pca = PCA(n_components=16)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[44]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[46]:


explained_variance = np.cumsum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
# Plotting the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(0, 26), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()


# In[47]:


variance_threshold = 0.95
n_components = np.argmax(explained_variance >= variance_threshold) + 1
print(f"Number of components to explain {variance_threshold * 100}% variance: {n_components}")


# In[48]:


s = pd.DataFrame(X_scaled)
s


# In[49]:


pca=PCA(n_components=7)


# In[50]:


pca.fit(X_scaled)


# In[51]:


x_pca=pca.transform(X_scaled)


# In[52]:


x_pca.shape


# In[53]:


m = pd.DataFrame(x_pca)


# In[55]:


m.columns = [f'Component_{i+1}' for i in range(7)]
m


# In[56]:


m.to_csv('PCA_add_7.csv', index=False)


# In[57]:


n = 156
new_df = m.iloc[n:]  #Comp_1 and Comp_2 for Cu_Pred
L = pd.DataFrame(new_df)
L

