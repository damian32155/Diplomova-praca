
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

orders = pd.read_csv('order_products2.csv')
products = pd.read_csv('products.csv')

order_products = pd.merge(orders, products, on='product_id')
order_products.rename(columns={'order_id':'order', 'product_name':'items'},inplace=True)
order_products['temp']=1
df_encoder = order_products.groupby(['order','items'])['temp'].sum().unstack().fillna(0)

frequent_itemsets = apriori(df_encoder, min_support=0.009, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Create graph
G = nx.DiGraph()

# add nodes
for item in rules['antecedents']:
    G.add_node(tuple(item), node_color='red', node_size=500)
for item in rules['consequents']:
    G.add_node(tuple(item), node_color='green', node_size=250)

# add edges
for index, row in rules.iterrows():
    G.add_edge(tuple(row['antecedents']), tuple(row['consequents']), weight=row['support'])

# plot network graph
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=[n[1]['node_color'] for n in G.nodes(data=True)], node_size=[n[1]['node_size'] for n in G.nodes(data=True)])
nx.draw_networkx_edges(G, pos, width=[d['weight']*10 for (u,v,d) in G.edges(data=True)])
nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.show()