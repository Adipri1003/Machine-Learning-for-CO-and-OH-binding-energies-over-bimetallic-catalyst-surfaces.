#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # Removing features that very low in feature importance scale consistently.

# In[16]:


df= pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
NC = df.iloc[:, 15]


# In[17]:


df2=df.iloc[:,6:9]
df2
df2 = df.drop(columns=['Element','CO_B.E','M-M.P', 'M-B.P', 'M-Enth.fus', 'M-Surface.E', 'M-1st Ion E', 'M-Group', 'M-Work F.', 'M.P', 'Enth.fus', 'At.radii']) # Replace with the actual column names
df2


# In[22]:


print(df2.columns.tolist())


# import pandas as pd
# 
# # Load dataset
# df = pd.read_csv('DATASET_A3B_CO-without_Cu-Copy1.csv')
# 
# # Check for NaN values in the entire dataset
# nan_summary = df.isna().sum()
# 
# # Display columns that contain NaN values
# nan_columns = nan_summary[nan_summary > 0]
# print("Columns with NaN values and their count:\n", nan_columns)
# 
# # Find rows where NaN values are present
# nan_rows = df[df.isna().any(axis=1)]
# print("\nRows with NaN values:\n", nan_rows)

# In[23]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2)


# In[25]:


pca = PCA(n_components=26)  # You can start by setting the number of components equal to the original feature set
pca.fit(X_scaled)


# In[26]:


X_pca = pca.fit_transform(X_scaled)  # Perform both fitting and transformation

# Retrieve cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[27]:


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


# In[28]:


variance_threshold = 0.95
n_components = np.argmax(explained_variance >= variance_threshold) + 1
print(f"Number of components to explain {variance_threshold * 100}% variance: {n_components}")


# In[29]:


s = pd.DataFrame(X_scaled)
s


# In[30]:


pca=PCA(n_components=9)


# In[31]:


pca.fit(X_scaled)


# In[32]:


x_pca=pca.transform(X_scaled)


# In[33]:


x_pca.shape


# In[34]:


m = pd.DataFrame(x_pca)


# In[36]:


m.columns = [f'Component_{i+1}' for i in range(9)]
m


# In[37]:


m.to_csv('PCA_add_9.csv', index=False)


# In[38]:


n = 156
new_df = m.iloc[n:]  #Comp_1 and Comp_2 for Cu_Pred
L = pd.DataFrame(new_df)
L


# In[40]:


L.to_csv('PCA_add_Cu_Pred_9.csv', index=False)


# In[ ]:





# Positively correlated
# 

# In[ ]:





# In[ ]:




