
# coding: utf-8

# In[54]:

import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import seaborn as sns
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().magic(u'matplotlib inline')


# In[55]:

train_df = pd.read_csv('forest_train.csv')


# In[56]:

soil_types = pd.read_csv('soil_types.csv').set_index('Soil Type')


# In[58]:

def labelSoilType(row):
    for i in range(len(row)):
        if row[i] == 1:
            return 'Soil_Type'+str(i)


# In[59]:

train_df['Soil Type'] = train_df[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
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
train_df = train_df.set_index('Soil Type')


# In[60]:

new_df = pd.merge(train_df, soil_types, how='inner', left_index=True, right_index=True)


# In[61]:

normalize(train_df, norm='l1', copy=False, axis=0)


# In[62]:

train_df.describe()


# In[63]:

train_df.columns


# In[64]:

train_df[['Aspect', 'Elevation', 'Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm',  
    'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology', 
    'Horizontal_Distance_To_Roadways', 'Slope', 'Vertical_Distance_To_Hydrology']].hist(alpha=0.7, figsize=(22, 14), bins=25, color='navy', layout=(5, 5))
plt.show()


# In[65]:

grouped_sum = pd.DataFrame(train_df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']].sum())

grouped_count = pd.DataFrame(train_df[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']].count())

grouped = pd.concat([grouped_sum, grouped_count], axis=1)
grouped.columns = ['Sum', 'Count']
grouped['1'] = grouped['Sum'] / grouped['Count']
grouped[['1']].plot(kind='bar', alpha=0.7, color='navy', figsize=(16, 8), legend=False)
# grouped.plot(kind='bar')


# In[66]:

continuous_fields = ['Aspect', 'Elevation', 'Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm',  
    'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology', 
    'Horizontal_Distance_To_Roadways', 'Slope', 'Vertical_Distance_To_Hydrology']
for i in continuous_fields:
    for j in continuous_fields:
        if i != j:
            plt.scatter(train_df[[i]], train_df[[j]], alpha=0.1)
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()


# In[67]:

train_df[continuous_fields].plot(kind='box', figsize=(20, 8), rot=45)
plt.show()


# In[68]:

fig, ax = plt.subplots(figsize = (20, 20))
ax.matshow(train_df.corr())
plt.xticks(range(len(train_df.corr().columns)), train_df.corr().columns, rotation=50);
plt.yticks(range(len(train_df.corr().columns)), train_df.corr().columns);


# In[69]:

X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Cover_Type', 1), train_df['Cover_Type'])


# In[71]:

model = SVC()


# In[72]:

model.fit(X_train, y_train)


# In[73]:

model.predict(X_test)


# In[74]:

model.score(X_test, y_test)


# In[ ]:

test_df = pd.read_csv('forest_test.csv')


# In[ ]:

X_test = test_df.drop('Cover_Type', 1)
y_test = test_df['Cover_Type']

