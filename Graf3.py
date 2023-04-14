
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
fig, ax = plt.subplots()
ax.scatter(rules['lift'], rules['confidence'], alpha=0.5, color='yellow')
ax.plot([1, 3.2], [0.13, 0.22], 'r-')
ax.set_xlabel('lift')
ax.set_ylabel('Confidence')
ax.set_title('lift vs Confidence')
plt.show()