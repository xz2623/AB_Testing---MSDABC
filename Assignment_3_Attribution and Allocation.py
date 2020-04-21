#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


#import data
attribution_data = pd.read_csv('attribution_allocation_student_data.csv')
channel_spend = pd.read_csv('channel_spend_student_data.csv')

tier1_spending_dict = {'social': 50, 'organic_search': 0, 'referral': 50, 'email': 50, 'paid_search': 50, 'display': 50, 'direct': 0}
tier2_spending_dict = {'social': 100, 'organic_search': 0, 'referral': 100, 'email': 100, 'paid_search': 100, 'display': 100, 'direct': 0}
tier3_spending_dict = {'social': 150, 'organic_search': 0, 'referral': 150, 'email': 150, 'paid_search': 150, 'display': 150, 'direct': 0}
total_spending_dict = {'social': 300, 'organic_search': 0, 'referral': 300, 'email': 300, 'paid_search': 300, 'display': 300, 'direct': 0}

#extract conversion data
convert_df = attribution_data.loc[attribution_data.convert_TF==True]


# In[7]:


value = []
for record in convert_df.tier:
    if record==1:
        value.append(50)
    elif record==2:
        value.append(100)
    else:
        value.append(150)


# In[8]:


convert_df['value']=value


# In[9]:


def channel_extraction(df,index):
    records = df.iloc[index,].isna()
    data =  convert_df.iloc[index,]
    channels = []
    for i in range(1,6):
        if records[i]!=1:
            channels.append(data[i])
    return channels


# In[10]:


channels_info=[]
for index in range(len(convert_df)):
    channels_info.append(channel_extraction(convert_df,index))


# # Attribution

# In[18]:


#last_interaction
last_interaction = [record[-1] for record in channels_info] 


# In[19]:


#last_non_click model
last_non_click=[]
for record in channels_info:
    try:
        if record[-1]!='direct':
            last_non_click.append(record[-1])
        else:
            last_non_click.append(record[-2])
    except:
        last_non_click.append(None)


# In[20]:


convert_df['last_interaction']=last_interaction
#convert_df['first_interaction']=first_interaction
convert_df['last_non_click']=last_non_click


# In[21]:


# split tiers
tier1_df = convert_df.loc[convert_df.tier==1]
tier2_df = convert_df.loc[convert_df.tier==2]
tier3_df = convert_df.loc[convert_df.tier==3]


# In[22]:


def attribution_model(method,df,tier_dict):
    df_1 = df[['value',method]]
    gb_df = df_1 .groupby(method).agg(len)

    gb_df['cost']=[tier_dict[record] for record in gb_df.index]
    gb_df['CAC'] = gb_df.cost/gb_df.value
    return gb_df


# ### Tier 1

# In[23]:


#tier1_last_interaction
tier1_last_inter = attribution_model('last_interaction',tier1_df,tier1_spending_dict)


# In[133]:


tier1_last_inter.sort_values('CAC',ascending=False)


# In[143]:


#tier1_first_interaction
tier1_first_inter = attribution_model('first_interaction',tier1_df,tier1_spending_dict)


# In[138]:


tier1_first_inter.sort_values('CAC',ascending=False)


# In[24]:


#tier1_last_nonclick
tier1_non_click = attribution_model('last_non_click',tier1_df,tier1_spending_dict)


# In[25]:


tier1_non_click.sort_values('CAC',ascending=False)


# ### Tier2

# In[26]:


#tier2_last_interaction
tier2_last_inter = attribution_model('last_interaction',tier2_df,tier2_spending_dict)


# In[27]:


tier2_last_inter.sort_values('CAC',ascending=False)


# In[28]:


#tier2_last_nonclick
tier_non_click = attribution_model('last_non_click',tier2_df,tier2_spending_dict)


# In[29]:


tier_non_click.sort_values('CAC',ascending=False)


# ## Tier3

# In[30]:


tier3_last_inter = attribution_model('last_interaction',tier3_df,tier3_spending_dict)


# In[31]:


tier3_last_inter.sort_values('CAC',ascending=False)


# In[32]:


tier3_non_click = attribution_model('last_non_click',tier3_df,tier3_spending_dict)
tier3_non_click.sort_values('CAC',ascending=False)


# ## total

# In[33]:


attribution_model('last_interaction',convert_df,total_spending_dict).sort_values('CAC',ascending=False)


# In[34]:


attribution_model('last_non_click',convert_df,total_spending_dict).sort_values('CAC',ascending=False)


# In[35]:


# linear model
def linear_model(channel_list,spending_dict):
    linear_dict = {k:0 for k in total_spending_dict.keys()}
    for record in channel_list:
        for channels in record:
            for channel in linear_dict.keys():
                if channel==channels:
                    linear_dict[channel]+=(1/len(record))
    linear_df = pd.DataFrame(linear_dict,index=linear_dict.keys()).T
    linear_df=linear_df[['social']]
    linear_df.columns = ['numbers']
    linear_df['cost']=[spending_dict[record] for record in linear_df.index]
    linear_df['CAC']=linear_df.cost/linear_df.numbers
    return linear_df


# In[36]:


#linear model
## total  linear model
total_linear_df=linear_model(channels_info,total_spending_dict)


# In[37]:


total_linear_df.sort_values('CAC',ascending=False)


# In[39]:


## tier1 linear models
channels_1=[channel_extraction(tier1_df,index) for  index in range(len(tier1_df))]
tier1_linear_df=linear_model(channels_1,tier1_spending_dict)
tier1_linear_df.sort_values('CAC',ascending=False)


# In[40]:


## tier2 linear models
channels_2=[channel_extraction(tier2_df,index) for  index in range(len(tier2_df))]
tier2_linear_df=linear_model(channels_2,tier2_spending_dict)
tier2_linear_df.sort_values('CAC',ascending=False)


# In[41]:


## tier3 linear models
channels_3=[channel_extraction(tier3_df,index) for  index in range(len(tier3_df))]
tier3_linear_df=linear_model(channels_3,tier3_spending_dict)
tier3_linear_df.sort_values('CAC',ascending=False)


# # Allocation

# In[52]:


allocation_df = pd.pivot_table(convert_df,index='last_non_click',columns='tier',aggfunc=len).convert_TF


# In[57]:


allocation_df['margin_acquision1']=allocation_df[1]
allocation_df['margin_acquision2']=allocation_df[2]-allocation_df[1]
allocation_df['margin_acquision3']=allocation_df[3]-allocation_df[2]


# In[59]:


allocation_df['marginal_CAC_1']=50/allocation_df.margin_acquision1
allocation_df['marginal_CAC_2']=50/allocation_df.margin_acquision2
allocation_df['marginal_CAC_3']=50/allocation_df.margin_acquision3


# In[67]:


allocation_df


# In[63]:


margin_CAC_df = allocation_df[[ 'marginal_CAC_1',    'marginal_CAC_2',    'marginal_CAC_3']]


# In[66]:


margin_CAC_df.sort_values('marginal_CAC_1',ascending=False)

