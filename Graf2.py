
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

orders = pd.read_csv('order_products2.csv')
products = pd.read_csv('products.csv')

order_products = pd.merge(orders, products, on='product_id')
order_products.rename(columns={'order_id':'order', 'product_name':'items'},inplace=True)
order_products['temp']=1
df_encoder = order_products.groupby(['order','items'])['temp'].sum().unstack().fillna(0)

frequent_itemsets = apriori(df_encoder, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Plot the support of the frequent itemsets
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()