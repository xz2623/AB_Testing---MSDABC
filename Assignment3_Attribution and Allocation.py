#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[3]:


#import data
attribution_data = pd.read_csv('attribution_allocation_student_data.csv')
channel_spend = pd.read_csv('channel_spend_student_data.csv')

tier1_spending_dict = {'social': 50, 'organic_search': 0, 'referral': 50, 'email': 50, 'paid_search': 50, 'display': 50, 'direct': 0}
tier2_spending_dict = {'social': 100, 'organic_search': 0, 'referral': 100, 'email': 100, 'paid_search': 100, 'display': 100, 'direct': 0}
tier3_spending_dict = {'social': 150, 'organic_search': 0, 'referral': 150, 'email': 150, 'paid_search': 150, 'display': 150, 'direct': 0}
total_spending_dict = {'social': 300, 'organic_search': 0, 'referral': 300, 'email': 300, 'paid_search': 300, 'display': 300, 'direct': 0}

#extract conversion data
convert_df = attribution_data.loc[attribution_data.convert_TF==True]


# In[12]:


value = []
for record in convert_df.tier:
    if record==1:
        value.append(50)
    elif record==2:
        value.append(100)
    else:
        value.append(150)


# In[13]:


convert_df['value']=value


# In[233]:


def channel_extraction(df,index):
    records = df.iloc[index,].isna()
    data =  convert_df.iloc[index,]
    channels = []
    for i in range(1,6):
        if records[i]!=1:
            channels.append(data[i])
    return channels


# In[68]:


channels_info=[]
for index in range(len(convert_df)):
    channels_info.append(channel_extraction(convert_df,index))


# In[71]:


#two attribution models
last_interaction = [record[-1] for record in channels_info] 
first_interaction = [record[0] for record in channels_info]


# In[83]:


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


# In[180]:


linear_model_dict


# In[88]:


convert_df['last_interaction']=last_interaction
convert_df['first_interaction']=first_interaction
convert_df['last_non_click']=last_non_click


# In[89]:


# split tiers
tier1_df = convert_df.loc[convert_df.tier==1]
tier2_df = convert_df.loc[convert_df.tier==2]
tier3_df = convert_df.loc[convert_df.tier==3]


# In[172]:


def attribution_model(method,df,tier_dict):
    df_1 = df[['value',method]]
    gb_df = df_1 .groupby(method).agg(len)

    gb_df['cost']=[tier_dict[record] for record in gb_df.index]
    gb_df['CAC'] = gb_df.cost/gb_df.value
    return gb_df


# ### Tier 1

# In[132]:


#tier1_last_interaction
tier1_last_inter = attribution_model('last_interaction',tier1_df,tier1_spending_dict)


# In[133]:


tier1_last_inter.sort_values('CAC',ascending=False)


# In[143]:


#tier1_first_interaction
tier1_first_inter = attribution_model('first_interaction',tier1_df,tier1_spending_dict)


# In[138]:


tier1_first_inter.sort_values('CAC',ascending=False)


# In[141]:


#tier1_last_nonclick
tier1_non_click = attribution_model('last_non_click',tier1_df,tier1_spending_dict)


# In[142]:


tier1_non_click.sort_values('CAC',ascending=False)


# ### Tier2

# In[152]:


#tier2_last_interaction
tier2_last_inter = attribution_model('last_interaction',tier2_df,tier2_spending_dict)


# In[153]:


tier2_last_inter.sort_values('CAC',ascending=False)


# In[154]:


#tier2_first_interaction
tier2_first_inter = attribution_model('last_interaction',tier2_df,tier2_spending_dict)


# In[155]:


tier2_first_inter.sort_values('CAC',ascending=False)


# In[156]:


#tier2_last_nonclick
tier_non_click = attribution_model('last_non_click',tier2_df,tier2_spending_dict)


# In[157]:


tier_non_click.sort_values('CAC',ascending=False)


# ## Tier3

# In[160]:


tier3_last_inter = attribution_model('last_interaction',tier3_df,tier3_spending_dict)


# In[161]:


tier3_last_inter.sort_values('CAC',ascending=False)


# In[162]:


tier3_first_inter = attribution_model('first_interaction',tier3_df,tier3_spending_dict)


# In[163]:


tier3_first_inter.sort_values('CAC',ascending=False)


# In[164]:


tier3_non_click = attribution_model('last_non_click',tier3_df,tier3_spending_dict)
tier3_non_click.sort_values('CAC',ascending=False)


# ## total

# In[166]:


convert_df.columns


# In[173]:


attribution_model('last_interaction',convert_df,total_spending_dict).sort_values('CAC',ascending=False)


# In[174]:


attribution_model('first_interaction',convert_df,total_spending_dict).sort_values('CAC',ascending=False)


# In[175]:


attribution_model('last_non_click',convert_df,total_spending_dict).sort_values('CAC',ascending=False)


# In[260]:


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


# In[261]:


#linear model
## total  linear model
total_linear_df=linear_model(channels_info,total_spending_dict)


# In[262]:


total_linear_df.sort_values('CAC',ascending=False)


# In[264]:


## tier1 linear models
channels_1=[channel_extraction(tier1_df,index) for  index in range(len(tier1_df))]
tier1_linear_df=linear_model(channels_1,tier1_spending_dict)
tier1_linear_df.sort_values('CAC',ascending=False)


# In[265]:


## tier2 linear models
channels_2=[channel_extraction(tier2_df,index) for  index in range(len(tier2_df))]
tier2_linear_df=linear_model(channels_2,tier2_spending_dict)
tier2_linear_df.sort_values('CAC',ascending=False)


# In[266]:


## tier3 linear models
channels_3=[channel_extraction(tier3_df,index) for  index in range(len(tier3_df))]
tier3_linear_df=linear_model(channels_3,tier3_spending_dict)
tier3_linear_df.sort_values('CAC',ascending=False)

