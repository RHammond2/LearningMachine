
# coding: utf-8

# In[16]:

import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.tools.plotting import scatter_matrix
import numpy as np
get_ipython().magic(u'matplotlib inline')


# In[17]:

df = pd.read_csv('train.csv', sep=',', header=0, index_col='Id')
df.head()


# In[18]:

soil_types = pd.read_csv('LearningMachine/soil_types.csv').set_index('Soil Type')


# In[19]:

def labelSoilType(row):
    for i in range(len(row)):
        if row[i] == 1:
            return 'Soil_Type'+str(i)


# In[20]:

df['Soil Type'] = df[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']].apply(lambda row: labelSoilType(row), axis=1)


# In[21]:

df = pd.merge(df, soil_types, how='left', left_on='Soil Type', right_index=True)


# In[22]:

for i in range(df.shape[1]-2, 14, -1):
    if df[df.columns.tolist()[i]].sum() == 0:
        df = df.drop(df.columns.tolist()[i], 1)


# In[23]:

sns.countplot(df.Cover_Type)


# In[24]:

plt.figure(figsize=(10,5))
plt.hist(df.Horizontal_Distance_To_Fire_Points, bins=20)
plt.title('Horizontal Distance to Fire Points Distribution')


# In[25]:

plt.figure(figsize=(10,5))
plt.hist(df.Horizontal_Distance_To_Hydrology, bins=20)
plt.title('Horizontal Distance to Hydrology')


# In[26]:

plt.figure(figsize=(10,5))
plt.hist(df.Vertical_Distance_To_Hydrology, bins=20)
plt.title('Vertical Distance to Hydrology')


# In[27]:

df.describe().transpose()


# In[28]:

def azimuth_to_abs(x):
    if x>180:
        return 360-x
    else:
        return x

df['Aspect2'] = df.Aspect.map(azimuth_to_abs)


# In[29]:

df['Above_Sealevel'] = (df.Vertical_Distance_To_Hydrology>0).astype(int)


# In[30]:

plt.figure(figsize=(15,5))
plt.hist(df.Elevation, bins=100)
plt.xticks(np.arange(1800, 4000, 100))
plt.xlim(1800, 3900)
plt.title('Elevation looks tri-modal- lets bucket that bish')


# In[31]:

bins = [0, 2600, 3100, 8000]
group_names = [1, 2, 3]


# In[32]:

df['Elevation_Bucket'] = pd.cut(df['Elevation'], bins, labels=group_names)


# In[33]:

plt.figure(figsize=(10,5))
plt.hist(df.Aspect, bins=100)
plt.title('Aspect')


# In[34]:

plt.figure(figsize=(7,7))
plt.scatter(df.Hillshade_9am, df.Hillshade_3pm)
plt.xlabel('Hillshade 9am')
plt.ylabel('Hillshade 3pm')


# In[35]:

plt.figure(figsize=(7,7))
plt.scatter(df.Hillshade_9am, df.Hillshade_Noon)
plt.xlabel('Hillshade 9am')
plt.ylabel('Hillshade Noon')


# In[36]:

plt.figure(figsize=(7,7))
plt.scatter(df.Hillshade_Noon, df.Hillshade_3pm)
plt.xlabel('Hillshade Noon')
plt.ylabel('Hillshade 3PM')


# In[37]:

sns.countplot(df[df.Hillshade_3pm==0].Cover_Type)
plt.title('0 Hillshade at 3PM Cover Types')


# In[38]:

plt.hist(df[df.Hillshade_3pm==0].Aspect, bins=50)
plt.title('Hist of Aspect for 3PM 0 Hillshade')


# In[39]:

df['3PM_0_Hillshade'] = (df.Hillshade_3pm==0).astype(int)


# In[40]:

plt.hist(df.Aspect, bins=50)
plt.title('Hist of Aspect All')


# In[41]:

axs=scatter_matrix(df[df.columns.tolist()[0:10]], alpha=0.2, figsize=(16, 16), diagonal='kde')
labelpads=[20, 20, 20, 85, 85, 85, 40, 40, 40, 85]
for x in range(10):
    for y in range(10):
        ax = axs[x, y]
        ax.xaxis.label.set_rotation(60)
        ax.yaxis.label.set_rotation(30)
        ax.yaxis.labelpad = labelpads[x]


# In[42]:

soil_types= ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40', 'Cover_Type']


# In[43]:

column_list = df.columns.tolist()
del column_list[14:54]
column_list.insert(1, 'Elevation_Bucket')
column_list.insert(3, 'Aspect2')
column_list.insert(6, 'Above_Sealevel')
column_list.insert(11, '3PM_0_Hillshade')
column_list=column_list[:-4]
column_list.extend(soil_types)


# In[44]:

df = df[column_list]


# In[45]:

for i in range(df.shape[1]-1):
    for j in range(54):
        if i != j:
            df[df.columns.tolist()[i]+"_"+df.columns.tolist()[j]] = df[df.columns.tolist()[i]]*df[df.columns.tolist()[j]]


# In[46]:

df.shape


# In[ ]:

for i in range(df.shape[1]-2, 92, -1):
    if df[df.columns.tolist()[i]].sum() == 0:
        df = df.drop(df.columns.tolist()[i], 1)


# In[ ]:




# Take top X variables look at interactions and correlation with label
# 
# Look at the covariate plots with the label coded in color to find banding or regions of patterns

# In[ ]:




# In[ ]:



