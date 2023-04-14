import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

orders = pd.read_csv('order_products2.csv')
products = pd.read_csv('products.csv')

order_products = pd.merge(orders, products, on='product_id')
order_products.rename(columns={'order_id':'order', 'product_name':'items'},inplace=True)
order_products['temp']=1
df_encoder = order_products.groupby(['order','items'])['temp'].sum().unstack().fillna(1)


frequent_itemsets = apriori(df_encoder, min_support=0.03, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

# Generate association rules with minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


# Create a bar plot
plt.bar(range(len(frequent_itemsets)), frequent_itemsets['support'], color='blue')

# Add labels to x and y axes
plt.xticks(range(len(frequent_itemsets)))
plt.xlabel('Frequent Itemsets')
plt.ylabel('Počet')

# Set the title of the plot
plt.title('Support Values of Frequent Itemsets')

# Show the plot
plt.show()