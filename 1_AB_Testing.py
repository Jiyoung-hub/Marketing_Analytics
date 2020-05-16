#!/usr/bin/env python
# coding: utf-8

# ## Read Data

# In[451]:


import pandas as pd


# In[452]:


path1 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\customer_service_reps'
path2 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\engagement'
path3 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\subscribers'
path4 = r'C:\Users\hahas\Desktop\Marketing Analytics\Final Case\advertising_spend_data.xlsx'

customer = pd.read_pickle(path1)
engagement = pd.read_pickle(path2)
subscribers = pd.read_pickle(path3)
ads = pd.read_excel(path4, skiprows=2)


# ## Limiting the Scope of Analysis

# In[453]:


# limit data scope to subscribers dataset excluding 'join_fee' Null
print("Total unique users in subscribers dataset: "+str(subscribers.subid.nunique())) # Total unique customers in subscribers dataset
print("Users with Null Value: "+str(subscribers.join_fee.isnull().sum())) # Null Valus
print("Excluding Nulls, the number of users in our focus: "+str(subscribers.join_fee.value_counts().sum())) # Our focus in A/B Testing


# In[454]:


# Check a distribution of 'join fee'
subscribers.join_fee.value_counts()


# In[455]:


# Variant_A: pay | Variant_B: free
free = subscribers[subscribers.join_fee <= 0]
pay = subscribers[subscribers.join_fee > 0]


# In[456]:


# Users in Variant_A and Variant_B
print("The number of users in Variant A is "+str(pay.subid.nunique()))
print("The number of users in Variant B is "+str(free.subid.nunique()))
print("Total users in AB testing is "+str(free.subid.nunique()+pay.subid.nunique())) # total


# ## Taking a look at Variant_A and Variant_B

# In[457]:


# Compare Retarget_TF ratio between Variant A and Variant B
free.retarget_TF.value_counts() / len(free)*100


# In[458]:


pay.retarget_TF.value_counts() / len(pay) * 100


# In[459]:


# Compare paid_TF ratio between Variant A and Variant B
free.paid_TF.value_counts() / len(free)*100


# In[460]:


# Compare paid_TF ratio between Variant A and Variant B
pay.paid_TF.value_counts() / len(pay) * 100


# In[461]:


# Compare operating system ratio
free.op_sys.value_counts() / len(free)*100


# In[462]:


pay.op_sys.value_counts() / len(pay) * 100


# In[463]:


# Compare plan type ratio
free.plan_type.value_counts() / len(free)*100


# In[464]:


pay.plan_type.value_counts() / len(pay) * 100


# In[465]:


# Compare 'discounted price' distribution
free.discount_price.value_counts() / len(free)*100


# In[466]:


pay.discount_price.value_counts() / len(pay) * 100


# In[467]:


# Compare the ratio of trials cancellation
free.cancel_before_trial_end.value_counts() / len(free)*100


# In[468]:


pay.cancel_before_trial_end.value_counts() / len(pay) * 100


# In[469]:


# Comapre payment type 
free.payment_type.value_counts() / len(free)*100


# In[470]:


pay.payment_type.value_counts() / len(pay) * 100


# In[471]:


# Comapre 'refund after trial_TF' ratio
free.refund_after_trial_TF.value_counts() / len(free)*100


# In[472]:


pay.refund_after_trial_TF.value_counts() / len(pay) * 100


# ## AB Testing 

# ### Calculating the optimal sample size

# In[473]:


import pandas as pd
from scipy.stats import norm
alpha = 0.05
power = 0.8


# In[474]:


# calculating p0
p0 = pay.paid_TF.sum() / pay.shape[0]
p0


# In[475]:


# calculating p1
p1 = free.paid_TF.sum() / free.shape[0]
p1


# In[476]:


# calculating delta
delta = (p1-p0)
delta


# In[477]:


# alpha
norm.ppf(1-alpha/2)


# In[478]:


# power
norm.ppf(1-power)


# In[479]:


# calculating the optimal sample size
from numpy import sqrt
alpha = 0.05
power = 0.8
t_alpha_d2 = norm.ppf(1-alpha/2)
t_beta = .84162
p0 = pay.paid_TF.sum() / pay.shape[0]
p1 = free.paid_TF.sum() / free.shape[0]
p_bar = (p0  + p1)/2


# In[480]:


# Optimal Sample Size: 2,946
((t_alpha_d2*sqrt(2*p_bar*(1-p_bar)))+t_beta*sqrt((p0*(1-p0))+(p1*(1-p1))))**2/delta/delta


# ## Calculating Z-score

# In[481]:


# Z_score: one sample test
p_sample = free.paid_TF.sum() / free.shape[0]
p = pay.paid_TF.sum() / pay.shape[0]
z_score = (p_sample-p) / sqrt(p*(1-p)/free.shape[0])
z_score


# In[482]:


# z_score: two sample test
n2 = pay.shape[0]
n1 = free.shape[0]
p2 = pay.paid_TF.sum() / n2
p1 = free.paid_TF.sum() / n1
p = (p1 * n1 + p2 * n2) / (n1 + n2)
z_score = (p1-p2)/ sqrt(p*(1-p)*(1/n1+1/n2))
z_score


# ## Calculating z_score for 10 trials

# In[483]:


def z_score(data, number):
    
    p = pay.paid_TF.sum() / pay.shape[0]
    variant_B = data
    variant_B_sampled = variant_B.sample(n = number)
    p_sample = variant_B_sampled.paid_TF.sum() / number
    z_score = (p_sample-p) / sqrt(p*(1-p)/number)

    return p_sample, z_score


# In[484]:


z_score(free, 2946)


# In[485]:


reject_null = 0
accept_null = 0

for i in list(range(1,11)):
    if z_score(free,2946)[1] >= 1.96:
        reject_null += 1
    else:
        accept_null += 1


# In[486]:


reject_null


# In[487]:


accept_null


# ## Sequential Testing

# In[488]:


import numpy as np


# In[489]:


# Sequential Testing
# From Type I error = 5% and Type II error = 20%, calculate upper and lower bounds
# Upper = ln(1/α) = 2.99
# Lower = ln(β) = -1.6

def squential_test(data,number,p_ho,p_h1):
    
    variant_B = data
    variant_B_sampled = variant_B.sample(n = number)
    total_sum=0
    i=0

    while -1.6 < total_sum <  2.99:     
        if variant_B_sampled.paid_TF.iloc[i] == True:
            log = np.log(p_h1/p_ho)
        elif variant_B_sampled.paid_TF.iloc[i] == False:
            log = np.log((1-p_h1)/(1-p_ho))
        total_sum = total_sum + log
        i = i+1
    
    return i, total_sum


# In[517]:


p0 = pay.paid_TF.sum() / pay.shape[0]
p1 = free.paid_TF.sum() / free.shape[0]
squential_test(free, 5000, p_ho=p0, p_h1=p1)


# In[498]:


# Showing the process of sequential Test
import numpy as np
p_ho=p0
p_h1=p1
i=0
total_sum = 0
while -1.6 < total_sum < 2.99:
    if free.sample(n = 2946).paid_TF.iloc[i] == True:
        log = np.log(p_h1/p_ho)
    elif free.sample(n = 2946).paid_TF.iloc[i] == False:
        log = np.log((1-p_h1)/(1-p_ho))
    total_sum = total_sum + log
    print(i, total_sum)
    i = i+1

