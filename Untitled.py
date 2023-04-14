#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import missingno as msno

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
#time and warnings
import time
import warnings

#settings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('poster', font_scale=0.5)


# In[4]:


orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products_train = pd.read_csv('order_products__train.csv')


# In[5]:


def size64_to_size32(df):
    """
    It converts all 64-bit floats and integers to 32-bit
    """
    for c in df.columns:
        if df[c].dtypes == 'int64':
            df[c] = df[c].astype(np.int32)
        if df[c].dtypes == 'float64':
            df[c] = df[c].astype(np.float32)


# In[6]:


size64_to_size32(orders)
size64_to_size32(products)
size64_to_size32(aisles)
size64_to_size32(departments)
size64_to_size32(order_products_train)


# In[7]:


orders.head()


# In[8]:


orders.info()


# In[9]:


#exploring the number of missing values per feature in percentage
print('Počet chýbajúcich hodnôt: ', orders.isnull().values.sum())
print('Percento chýbajúcich hodnôt na objekt: ') 
orders.isnull().sum() * 100 / len(orders)


# In[10]:


msno.matrix(orders)


# In[19]:


products.head()


# In[20]:


products.info()


# In[21]:


print('Number of missing values: ', products.isnull().values.sum())
print('Percent of missing values per feature: ') 
products.isnull().sum() * 100 / len(products)


# In[22]:


aisles.head()


# In[23]:


aisles.info()


# In[24]:


departments.head()


# In[25]:


departments.info()


# In[26]:


#exploring the number of missing values per feature in percentage
print('Number of missing values: ', departments.isnull().values.sum())
print('Percent of missing values per feature: ') 
departments.isnull().sum() * 100 / len(departments)


# In[27]:


order_products_train.head()


# In[28]:


order_products_train.info()


# In[29]:


#exploring the number of missing values per feature in percentage
print('Number of missing values: ', order_products_train.isnull().values.sum())
print('Percent of missing values per feature: ') 
order_products_train.isnull().sum() * 100 / len(order_products_train)


# In[30]:


#merging products with aisles, departments, and order_products_train
products = pd.merge(aisles, products, on='aisle_id')
products = pd.merge(departments, products, on='department_id')
products = pd.merge(order_products_train, products, on='product_id')


# In[31]:


#merging products and orders
products_and_orders = pd.merge(products, orders, on='order_id')


# In[32]:


products.head()


# In[33]:


products_and_orders.head()


# In[34]:


#visualizing the number of customers, orders, aisles, and products
data = {'Customers': len(products_and_orders.user_id.unique()),
        'Orders': len(products_and_orders.order_id.unique()),
        'Products': len(products_and_orders.product_id.unique()),
        'Aisles': len(products_and_orders.aisle_id.unique()),
        'Departments': len(products_and_orders.department_id.unique())}
data_structure = pd.DataFrame(data, index=[0])

plt.figure(figsize=(12,6))
ax = sns.barplot(data = data_structure, palette = 'Set2')
ax.bar_label(ax.containers[0]);


# In[61]:


plt.figure(figsize=(12,8))
sns.countplot(orders['order_dow'])
plt.ylabel('Počet objednávok', fontsize=12)
plt.xlabel('Deň v týždni', fontsize=12)
plt.title('Počet objednávok za deň v týždni', fontsize=15)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Nedeľa', 'Sobota', 'Pondelok', 'Štvrtok', 'Piatok', 'Utorok', 'Streda'])
plt.show()


# In[38]:


#at what hour do people order the most

plt.figure(figsize=(15,8))
sns.countplot(orders['order_hour_of_day'])
plt.ylabel('Počet objednávok', fontsize=12)
plt.xlabel('Hodiny', fontsize=12)
plt.title('Počet objednávok za hodinu v dni', fontsize=15);


# In[65]:


#frequency of orders per weekday and its hour

days_hours = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
days_hours = days_hours.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(15,8))
sns.heatmap(days_hours, cmap='coolwarm')
plt.title('Frekvencia dňa v týždni a hodín dňa', fontsize=15)
plt.ylabel('Dni', fontsize = 12)
plt.xlabel('Hodiny', fontsize = 12)
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Nedeľa', 'Sobota', 'Pondelok', 'Štvrtok', 'Piatok', 'Utorok', 'Streda'])
plt.show()


# In[40]:


orders.groupby('user_id')['order_id'].nunique().describe()


# In[66]:


plt.figure(figsize=(12,8))
sns.histplot(orders.groupby('user_id')['order_id'].nunique(), binwidth=3, kde = True, palette = 'Set2')
plt.ylabel('Počet', fontsize=12)
plt.xlabel('Objednávky na zákazníka', fontsize=12)
plt.title('Počet objednávok na zákazníka', fontsize=15)
plt.show();


# In[67]:


plt.figure(figsize=(14,8))
sns.countplot(orders['days_since_prior_order'], palette = 'Set3')
plt.ylabel('Počet objednávok', fontsize=12)
plt.xlabel('Dni', fontsize=12)
plt.title('Frekvencia objednávok za dni', fontsize=15)
plt.show();


# In[69]:


plt.figure(figsize=(12,6))
sns.barplot(data = products.groupby('product_name')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:15], 
            x = 'add_to_cart_order', y = 'product_name', palette = 'Set2')
plt.ylabel('Produkty', fontsize=12)
plt.xlabel('Top 15 produktov pridaných do košíka', fontsize=12)
plt.title('Najčastejšie objednávané produkty', fontsize=13)
plt.show();


# In[71]:


#visualizing top 12 products that were reordered

plt.figure(figsize=(12,6))
sns.barplot(data = products.groupby('product_name')['reordered'].sum().sort_values(ascending = False).reset_index()[0:15], 
            x = 'reordered', y = 'product_name', palette = 'Set3')
plt.ylabel('Produkty', fontsize=12)
plt.xlabel('Počet opakovaní objednávky', fontsize=12)
plt.title('Top 15 preobjednaných produktov', fontsize=13)
plt.show();


# In[76]:


#visualizing the proportion of reordered items

prop_reorder = order_products_train.groupby('reordered')['add_to_cart_order'].count().reset_index()
prop_reorder['proportion'] = prop_reorder['add_to_cart_order'] * 100 / order_products_train['add_to_cart_order'].count()

plt.figure(figsize=(10,6))
sns.barplot(data = prop_reorder, x='reordered', y='proportion', palette='coolwarm')
plt.ylabel('Pomer', fontsize=12)
plt.xlabel('Preobjednávka', fontsize=0)
plt.title('Pomer preobjednaných poroduktov', fontsize=15)
plt.xticks(ticks=[0, 1], labels=['Preobjednané produkty', 'Objednané iba raz'])
plt.show();


# In[77]:


#visualizing how many products are there in the basket usually

products_per_order = order_products_train.groupby('order_id')['add_to_cart_order'].max().reset_index()

plt.figure(figsize=(14,8))
sns.histplot(order_products_train['add_to_cart_order'], bins = 50, kde = True)
plt.ylabel('Počet produktov', fontsize=12)
plt.xlabel('Objednávka produktov pridaných do košíka', fontsize=12)
plt.title('Počet produktov v košíku', fontsize=15)
plt.show();


# In[49]:


#visualizing most popular aisles per products ordered

plt.figure(figsize=(12,6))
sns.barplot(data = products_and_orders.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:10], 
            x = 'add_to_cart_order', y = 'aisle', palette = 'Set1')
plt.ylabel('Aisle', fontsize=12)
plt.xlabel('Products ordered', fontsize=12)
plt.title('Most popular Aisles', fontsize=13)
plt.show();


# In[50]:


top_aisle = products.groupby('aisle')['order_id'].count().reset_index()
top_aisle = top_aisle.nlargest(12,'order_id')

plt.figure(figsize=(12,6))
sns.barplot(data = top_aisle, x = 'order_id', y = 'aisle', palette = 'Set2')
plt.ylabel('Aisle', fontsize=12)
plt.xlabel('Number of orders', fontsize=12)
plt.title('Share of orders per aisle', fontsize=13)
plt.show();


# In[51]:


# most popular departments in terms of units ordered

plt.figure(figsize=(12,6))
sns.barplot(data = products_and_orders.groupby('department')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:10], 
            x = 'add_to_cart_order', y = 'department', palette = 'Set2')
plt.ylabel('Department', fontsize=12)
plt.xlabel('Products ordered', fontsize=12)
plt.title('Most popular Departments', fontsize=13)
plt.show();


# In[52]:


top_dep = products.groupby('department')['order_id'].count().reset_index()
top_dep = top_dep.nlargest(12,'order_id')

plt.figure(figsize=(12,6))
sns.barplot(data = top_dep, x = 'order_id', y = 'department', palette = 'Set3')
plt.ylabel('Department', fontsize=12)
plt.xlabel('Number of orders', fontsize=12)
plt.title('Share of orders per department', fontsize=13)
plt.show();


# In[ ]:




