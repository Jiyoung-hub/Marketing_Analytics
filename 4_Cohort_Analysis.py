#!/usr/bin/env python
# coding: utf-8

# ## Read data

# In[1]:


import pandas as pd


# In[2]:


path1 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\customer_service_reps'
path2 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\engagement'
path3 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\subscribers'
path4 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\advertising_spend_data.xlsx'

customer = pd.read_pickle(path1)
engagement = pd.read_pickle(path2)
subscribers = pd.read_pickle(path3)
ads = pd.read_excel(path4, skiprows=2)


# ## Cohort Analysis - customer dataset

# In[3]:


# Assign tier group
tier1_ = customer[customer.account_creation_date <= '2019-07-30']

tier2_ = customer[(customer.account_creation_date > '2019-07-30') & 
                    (customer.account_creation_date <= '2019-08-30')]

tier3_ = customer[(customer.account_creation_date > '2019-08-30') & 
                    (customer.account_creation_date <= '2019-09-30')]

tier4_ = customer[(customer.account_creation_date > '2019-09-30') & 
                    (customer.account_creation_date <= '2019-10-30')]

tier5_ = customer[(customer.account_creation_date > '2019-10-30') & 
                    (customer.account_creation_date <= '2019-11-30')]

tier6_ = customer[(customer.account_creation_date > '2019-11-30') & 
                    (customer.account_creation_date <= '2019-12-30')]

tier7_ = customer[(customer.account_creation_date > '2019-12-30') & 
                    (customer.account_creation_date <= '2020-01-30')]

tier8_ = customer[(customer.account_creation_date > '2020-01-30')& 
                    (customer.account_creation_date <= '2020-02-29')]

tier9_ = customer[(customer.account_creation_date > '2020-02-29') & 
                    (customer.account_creation_date <= '2020-03-30')]


# In[4]:


# create cohort table with number of users alive
cohort_table = pd.DataFrame()
tiers = [tier1_, tier2_,tier3_,tier4_,tier5_,tier6_,tier7_,tier8_,tier9_]
for tier in tiers:
    cohort = tier['payment_period'].value_counts()
    cohort_table = cohort_table.append(cohort)


# In[5]:


# reset index
s = pd.Series(['tier1','tier2','tier3','tier4','tier5','tier6','tier7','tier8','tier9'])
cohort_table.set_index(s)


# In[6]:


# Create cohort table with ratio of users alive
dic = {}
cohort_ratio = pd.DataFrame()

for i in list(range(0,9)):
    for j in list(range(0,13)):
        ratio = round(cohort_table.iloc[i][j] / cohort_table.iloc[i][0] * 100, 1)
        dic[j] = ratio
    df = pd.DataFrame(dic, index=['tier'+str(i)])
    cohort_ratio = cohort_ratio.append(df)

cohort_ratio


# In[7]:


# Columns names
columns = []
for i in list(range(0,13)):
    column = 'period ' + str(i)
    columns.append(column)


# In[8]:


cohort_ratio.columns = [columns]


# In[9]:


cohort_ratio

