#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rnd


# In[59]:


adjacency_list_file = np.loadtxt(fname='Adjlist_FR.tsv', delimiter='\t')
#adjacency_list_file = np.loadtxt(fname='graph_adjlist.tsv', delimiter='\t')


# In[60]:


GRAPH_POPULATION_SIZE = 50

population = [i for i in range(0, 5000)]
random_nodes = rnd.sample(population, GRAPH_POPULATION_SIZE)

subset_test = adjacency_list_file
#subset_test = adjacency_list_file[np.r_[1000:2000]] # 'Adjlist_FR.tsv'
#subset_test = adjacency_list_file
adjacency_list = [tuple(i) for i in subset_test]
print(adjacency_list[0:10])


# In[62]:


G = nx.Graph()
G.add_edges_from(adjacency_list)


# In[55]:


plt.figure(num=None, figsize=(10, 10), dpi=600)
nx.draw(G, 
        node_size=50, 
        width=1.0,
        alpha=0.8, 
        node_color='#00b4d9') #, with_labels=True)
plt.savefig("graph_viz.png")
plt.show()


# In[7]:


nx.write_graphml(G,'so.graphml')


# In[63]:


nx.density(G)


# In[64]:


nx.degree_histogram(G)


# In[65]:


nx.degree(G)


# In[ ]:




