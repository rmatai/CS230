#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from datetime import datetime
import kragle as kg
import sys
import pandas as pd
import math
import numpy as np
import scipy.stats as st
import statsmodels.stats.api as sms
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import sklearn as sk

pd.set_option('display.float_format', '{:.0f}'.format)


# In[2]:


deepwalk_tSNE = np.loadtxt(fname='tSNE.txt', delimiter=' ')
deepwalk_labels = np.loadtxt(fname='labels.txt', delimiter=' ', dtype=str)


# In[3]:


print(deepwalk_tSNE[0:2])
print(deepwalk_labels[0:2])


# In[4]:


DISTINCT_PROFILE_IDS = deepwalk_tSNE[:,[0]].flatten()
print('Distinct profile_ids: %i\n' % (len(DISTINCT_PROFILE_IDS)))

deepwalk_tSNE_df = pd.DataFrame(deepwalk_tSNE)
deepwalk_labels_df = pd.DataFrame(deepwalk_labels)

deepwalk_tSNE_df.rename({0: 'profile_id',
                         1: 'x1',
                         2: 'x2'},
                        axis=1, inplace=True)

deepwalk_labels_df.rename({0: 'profile_id',
                           1: 'label'},
                          axis=1, inplace=True)

deepwalk_labels_df['profile_id'] = pd.to_numeric(deepwalk_labels_df['profile_id'])
deepwalk_labels_df['label'] = deepwalk_labels_df['label'].astype(str)


DISTINCT_VERTICALS = list(set(deepwalk_labels_df['label'].values))
UNKNOWN_VERTICAL_LABEL = 'unknown'
dict_vertical_enumeration = dict([(y,x+1) for x,y in enumerate(sorted(set(DISTINCT_VERTICALS)))])
dict_vertical_enumeration.update({UNKNOWN_VERTICAL_LABEL: len(DISTINCT_VERTICALS) + 1}) # Add a dummy entry
print(dict_vertical_enumeration)


# In[16]:


data = []

for distinct_profile_id in DISTINCT_PROFILE_IDS:
    tSNE_row = deepwalk_tSNE_df[deepwalk_tSNE_df['profile_id'] == distinct_profile_id]
    vertical_label = deepwalk_labels_df[deepwalk_labels_df['profile_id'] == distinct_profile_id].label
    
    if vertical_label.empty:
        vertical_label = UNKNOWN_VERTICAL_LABEL
            
    vertical_enumeration = dict_vertical_enumeration[vertical_label.values[0]]
        
    data.append([distinct_profile_id, vertical_enumeration, tSNE_row.x1.values[0], tSNE_row.x2.values[0]])
    
data_df = pd.DataFrame(data, columns = ['profile_id', 'vertical', 'x1', 'x2'])


# In[17]:


join_distinct_verticals = list(set(data_df['vertical'].values))
print(join_distinct_verticals)


# In[18]:


print(data_df.head(2))
print(data[0:2])


# In[67]:


def get_key(dict_vertical_enumeration, val): 
    for key, value in dict_vertical_enumeration.items(): 
         if val == value: 
            return key 
        
from random import randint
color = []
n = 40
for i in range(n):
    color.append('#%06X' % randint(0, 0xFFFFFF))

unique_vertical_enumerations = set(dict_vertical_enumeration.values())

fig, ax = plt.subplots(figsize=(18, 18))

for vertical in unique_vertical_enumerations:
    ax1 = ax.scatter(data_df[data_df['vertical'] == vertical].x1,
                     data_df[data_df['vertical'] == vertical].x2,
                     c = color[vertical],
                     label = get_key(dict_vertical_enumeration, vertical),
                     s = 100)
ax.legend()

"""
ax1 = data_df.plot(kind="scatter",
                   x='x1',
                   y='x2',
                   c='vertical',
                   colormap='viridis',
                  ax=ax)
"""
fig_name = 'clustering_result.png'
plt.savefig(fig_name)

ax.grid(True)

plt.show()


# In[ ]:




