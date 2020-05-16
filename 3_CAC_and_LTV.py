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


# In[3]:


# Check minimun account creation date
subscribers.account_creation_date.min()


# In[4]:


# Check maximun account creation date
subscribers.account_creation_date.max()


# ### (All) Calculate the number of users attributed to each channel

# In[5]:


# Assign tiers to all customers 
tier1_all = customer[customer.account_creation_date <= '2019-07-30']

tier2_all = customer[(customer.account_creation_date > '2019-07-30') & 
                    (customer.account_creation_date <= '2019-08-30')]

tier3_all = customer[(customer.account_creation_date > '2019-08-30') & 
                    (customer.account_creation_date <= '2019-09-30')]

tier4_all= customer[(customer.account_creation_date > '2019-09-30') & 
                    (customer.account_creation_date <= '2019-10-30')]

tier5_all = customer[(customer.account_creation_date > '2019-10-30') & 
                    (customer.account_creation_date <= '2019-11-30')]

tier6_all = customer[(customer.account_creation_date > '2019-11-30') & 
                    (customer.account_creation_date <= '2019-12-30')]

tier7_all = customer[(customer.account_creation_date > '2019-12-30') & 
                    (customer.account_creation_date <= '2020-01-30')]

tier8_all = customer[(customer.account_creation_date > '2020-01-30')& 
                    (customer.account_creation_date <= '2020-02-29')]

tier9_all = customer[(customer.account_creation_date > '2020-02-29') & 
                    (customer.account_creation_date <= '2020-03-30')]


# In[6]:


# Calculate total signups for all users 
ads_adjusted = pd.DataFrame()
signups = []
tiers = [tier1_all, tier2_all, tier3_all, tier4_all, tier5_all, tier6_all, tier7_all, tier8_all, tier9_all]

for tier in tiers:
    signup = tier.subid.nunique()
    signups.append(signup)
    
Total_signup = pd.DataFrame(signups,columns=['Total_signup'])
ads_adjusted= pd.concat([ads_adjusted,Total_signup],axis=1)
ads_adjusted.index = list(['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6', 'tier7', 'tier8', 'tier9'])
ads_adjusted


# ### (known) Calculate the number of users attributed to each channel - Technical attribution data is known

# In[7]:


# Assign tiers to subscribers data
tier1 = subscribers[subscribers.account_creation_date <= '2019-07-30']

tier2 = subscribers[(subscribers.account_creation_date > '2019-07-30') & 
                    (subscribers.account_creation_date <= '2019-08-30')]

tier3 = subscribers[(subscribers.account_creation_date > '2019-08-30') & 
                    (subscribers.account_creation_date <= '2019-09-30')]

tier4 = subscribers[(subscribers.account_creation_date > '2019-09-30') & 
                    (subscribers.account_creation_date <= '2019-10-30')]

tier5 = subscribers[(subscribers.account_creation_date > '2019-10-30') & 
                    (subscribers.account_creation_date <= '2019-11-30')]

tier6 = subscribers[(subscribers.account_creation_date > '2019-11-30') & 
                    (subscribers.account_creation_date <= '2019-12-30')]

tier7 = subscribers[(subscribers.account_creation_date > '2019-12-30') & 
                    (subscribers.account_creation_date <= '2020-01-30')]

tier8 = subscribers[(subscribers.account_creation_date > '2020-01-30')& 
                    (subscribers.account_creation_date <= '2020-02-29')]

tier9 = subscribers[(subscribers.account_creation_date > '2020-02-29') & 
                    (subscribers.account_creation_date <= '2020-03-30')]


# In[8]:


# non-paid: 'internal', 'discovery', 'organic', 'google_organic','facebook_organic', 'bing_organic', 'pinterest_organic'
# Calcualte the number of users attributed to each channel 
channels = ['facebook', 'email', 'search', 'brand sem intent google',
       'affiliate', 'email_blast', 'pinterest', 'referral']
tiers = [tier1, tier2, tier3, tier4, tier5, tier6, tier7, tier8, tier9]
acquisition = pd.DataFrame()

for tier in tiers:
    num_cust = pd.DataFrame(tier['attribution_technical'].value_counts()).T
    acquisition = acquisition.append(num_cust)
    
acquisition.index = list(['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6', 'tier7', 'tier8', 'tier9'])
acquisition_sub = acquisition[channels]
acquisition_sub


# ## Ads spend allocation

# In[9]:


# Drop index=9 because we do not have customers who signed up after 3/31/2020
subscribers.account_creation_date.max()
ads = ads.drop(index=9)


# In[10]:


# Calculate signups of which technical attribution is known
known_signup = pd.DataFrame(acquisition_sub.sum(axis=1),columns=['Sign ups we know data for'])
ads_adjusted = pd.concat([ads_adjusted,known_signup],axis=1)
ads_adjusted


# In[11]:


# Get the % of customers we know data for
ads_adjusted['% of customers we know'] = ads_adjusted['Sign ups we know data for'] / ads_adjusted['Total_signup']
ads_adjusted


# In[12]:


ads_spend_for_subscribers = ads.copy()
for i in list(range(0,9)):
    for channel in channels:
        ads_spend_for_subscribers[channel][i] = ads_spend_for_subscribers[channel][i] * ads_adjusted['% of customers we know'][i]


# In[13]:


ads_spend_for_subscribers


# ## CAC by tier by channel for all

# In[14]:


# CAC by tier by channel 
dic = {}
CAC_by_tier_by_channel = pd.DataFrame()
channels = ['facebook', 'email', 'search', 'brand sem intent google',
       'affiliate', 'email_blast', 'pinterest', 'referral']
for i in list(range(0,9)):
    for channel in channels:
        CAC = ads_spend_for_subscribers.iloc[i][channel]/acquisition.iloc[i][channel]
        dic[channel] = CAC
    df = pd.DataFrame(dic, index=['tier'+str(i)])
    CAC_by_tier_by_channel = CAC_by_tier_by_channel.append(df) 

CAC_by_tier_by_channel


# In[15]:


# add mean at the bottom
avg = pd.DataFrame(CAC_by_tier_by_channel.mean()).T
avg.index = ['mean']
CAC_by_tier_by_channel = CAC_by_tier_by_channel.append(avg)


# In[16]:


CAC_by_tier_by_channel


# ## CAC by tier by channel for paid users only

# In[17]:


paid = subscribers[subscribers.paid_TF==True]
paid.shape


# In[18]:


tier1_p = paid[paid.account_creation_date <= '2019-07-30']

tier2_p = paid[(paid.account_creation_date > '2019-07-30') & 
                    (paid.account_creation_date <= '2019-08-30')]

tier3_p = paid[(paid.account_creation_date > '2019-08-30') & 
                    (paid.account_creation_date <= '2019-09-30')]

tier4_p = paid[(paid.account_creation_date > '2019-09-30') & 
                    (paid.account_creation_date <= '2019-10-30')]

tier5_p = paid[(paid.account_creation_date > '2019-10-30') & 
                    (paid.account_creation_date <= '2019-11-30')]

tier6_p = paid[(paid.account_creation_date > '2019-11-30') & 
                    (paid.account_creation_date <= '2019-12-30')]

tier7_p = paid[(paid.account_creation_date > '2019-12-30') & 
                    (paid.account_creation_date <= '2020-01-30')]

tier8_p = paid[(paid.account_creation_date > '2020-01-30')& 
                    (paid.account_creation_date <= '2020-02-29')]

tier9_p = paid[(paid.account_creation_date > '2020-02-29') & 
                    (paid.account_creation_date <= '2020-03-30')]


# In[19]:


# Calculating the number of paid users acquired by tier by channel
channels = ['facebook', 'email', 'search', 'brand sem intent google',
       'affiliate', 'email_blast', 'pinterest', 'referral']

tiers = [tier1_p, tier2_p, tier3_p, tier4_p, tier5_p, tier6_p, tier7_p, tier8_p, tier9_p]
acquisition_p = pd.DataFrame()

for tier in tiers:
    num_cust = pd.DataFrame(tier['attribution_technical'].value_counts()).T
    acquisition_p = acquisition_p.append(num_cust)
    
acquisition_p.index = list(['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6', 'tier7', 'tier8', 'tier9'])
acquisition_sub_p = acquisition_p[channels]
acquisition_sub_p


# In[20]:


# CAC_by_tier_by_channel for paid users 
dic = {}
CAC_by_tier_by_channel_paid = pd.DataFrame()
channels = ['facebook', 'email', 'search', 'brand sem intent google',
       'affiliate', 'email_blast', 'pinterest', 'referral']
for i in list(range(0,9)):
    for channel in channels:
        CAC = ads_spend_for_subscribers.iloc[i][channel]/acquisition_p.iloc[i][channel]
        dic[channel] = CAC
    df = pd.DataFrame(dic, index=['tier'+str(i)])
    CAC_by_tier_by_channel_paid = CAC_by_tier_by_channel_paid.append(df) 

CAC_by_tier_by_channel_paid


# In[21]:


# add mean at the bottem
avg = pd.DataFrame(CAC_by_tier_by_channel_paid.mean()).T
avg.index = ['mean']
CAC_by_tier_by_channel_paid = CAC_by_tier_by_channel_paid.append(avg)


# In[22]:


CAC_by_tier_by_channel_paid


# ## Exports files

# In[23]:


CAC_by_tier_by_channel.to_csv('CAC_by_tier_by_channel.csv',index=False)


# In[24]:


CAC_by_tier_by_channel_paid.to_csv('CAC_by_tier_by_channel_paid.csv',index=False)

