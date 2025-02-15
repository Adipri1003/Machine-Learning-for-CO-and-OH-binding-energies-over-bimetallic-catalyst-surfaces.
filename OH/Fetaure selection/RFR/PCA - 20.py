#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[2]:


df= pd.read_csv('Dataset_A3B_OH_343.csv')


# In[3]:


df2=df.iloc[0:69,1:38]
df2 = df2.drop(columns=['M-Enth.atom', 
                        'M-Elec.-ve', 'M-Surface.E', 'M-Group','M-M.P', 
                        
                         'Elec.-ve', 'cova .radii', 'At.radii',
                        'Group',  'Work F.', 'OH_B.E','M-Elec.Aff','M-Work F.', 'M-1st Ion E','M-At No.', 'M-At wt.', 'M-Sp.ht Cap','Sp.ht Cap', 'B.P', 'Elec.Aff','Density',]) 
df2


# In[4]:


print(df2.columns.tolist())


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[6]:


pca = PCA(n_components=16)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[7]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[8]:


explained_variance = np.cumsum(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
# Plotting the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(0, 16), explained_variance, marker='o', linestyle='--')
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


pca=PCA(n_components=5)


# In[12]:


pca.fit(X_scaled)


# In[13]:


x_pca=pca.transform(X_scaled)


# In[14]:


x_pca.shape


# In[15]:


m = pd.DataFrame(x_pca)


# In[16]:


m.columns = [f'Component_{i+1}' for i in range(5)]
m


# In[17]:


m.to_csv('PCA_add_5.csv', index=False)


# In[18]:


n = 156
new_df = m.iloc[n:]  #Comp_1 and Comp_2 for Cu_Pred
L = pd.DataFrame(new_df)
L


# In[58]:


L.to_csv('PCA_add_Cu_Pred.csv', index=False)


# In[ ]:





# Positively correlated
# 

# In[ ]:





# In[ ]:




