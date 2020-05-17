#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import pickle
import numpy as np
from scipy.stats import norm
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from scipy import stats

# here we set up all columns
pd.set_option('max_columns', 64)


# In[6]:


datafile = open('engagement', 'rb')

eng = pickle.load(datafile)

datafile.close()


# In[7]:


#datafile = open('customer_service_reps', 'rb')
datafile = open('customer_service_reps_dr', 'rb')

cust_sev = pickle.load(datafile)

datafile.close()


# In[8]:


datafile = open('subscribers', 'rb')

sub = pickle.load(datafile)

datafile.close()


# In[302]:


ad_spend = pd.read_csv('advertising_spend.csv', index_col=0)


# # customer_service_reps

# #datafile = open('customer_service_reps', 'rb')
# datafile = open('customer_service_reps_dr', 'rb')
# 
# cust_sev = pickle.load(datafile)
# 
# datafile.close()

# In[17]:


cust_sev.head(5)


# In[18]:


cust_sev.shape


# In[19]:


cust_sev.columns


# In[20]:


cust_sev.info()


# In[21]:


cust_sev['revenue_net_1month'].unique()


# ## Descriptive Analysis

# In[22]:


# There are 1369360 users in total in the customer_service_reps data
cust_sev['subid'].nunique()


# In[23]:


cust_sev['num_trial_days'].unique()


# In[24]:


cust_sev['billing_channel'].unique()


# ### Data Questions

# <b>Cancel_date</b>: 
# There are 153 unique users in the customer_service_reps have cancel_date earlier than their account_creation_date

# In[25]:


# cust_sev.head(50)


# In[26]:


"""dq_a = cust_sev[cust_sev['cancel_date'].isna() == 0]"""


# In[27]:


"""dq_a = dq_a[ dq_a['cancel_date'] < dq_a['account_creation_date']]
dq_a[['subid','cancel_date','account_creation_date']].reset_index(drop=True).head()"""


# In[28]:


"""dq_a['billing_channel'].unique()"""


# <b>Null Value</b>: 
# We need to manually fill in the null value of the <b>renew</b> column for <b>all three channels</b>

# In[29]:


"""dq_b = cust_sev[cust_sev['billing_channel'] == 'google']
dq_b[['subid','billing_channel','last_payment','next_payment','payment_period','renew']].head(4)"""


# In[30]:


"""dq_c = cust_sev[cust_sev['billing_channel'] == 'OTT']
dq_c = dq_c[np.logical_and(dq_c['next_payment'].isna() == True, dq_c['payment_period'] == 0)]
dq_c[['subid','billing_channel','last_payment','next_payment','payment_period','renew']].head()"""


# In[31]:


"""dq_d = cust_sev[cust_sev['billing_channel'] == 'itunes']
dq_d = dq_d[np.logical_or(dq_d['next_payment'].isna() == True, dq_d['payment_period'] == 0)]
dq_d[['subid','billing_channel','last_payment','next_payment','payment_period','renew']].head()"""


# In[32]:


cust_sev.nunique()


# #### Solutions

# ## Preprocessing the data

# In[33]:


"""
def renew_filling(period, nxt):
    if str(nxt) == 'NaT':
        return(False)
    else:
        return(True)
"""


# In[34]:


# cust_sev['renew'] = cust_sev.apply(lambda row: renew_filling( row['payment_period'], row['next_payment'] ), axis = 1)


# In[35]:


# now that we've preprocessed the dataset a bit, how can we easily write it to a file?
# let us use pickle to do that in 2-3 easy steps

# 1st: open a file in "write binary" (wb) mode
datafile = open("customer_service_reps_dr", "wb")

# 2nd: call pickle.dump(). First argument: the data structure or variable to be saved to the file
# Second argument: the file handle (or "file reference") opened above in "wb" mode

pickle.dump(cust_sev, datafile)

# 3rd: close that file to make sure everything was written to disk!
datafile.close()


# ## OTT Channel

# There are three billing channels \
# For <b>OTT channel</b>, there are only 2 trial days, <b>14 and 0</b> which we can use for <b>AB_Testing</b>

# In[9]:


ott = cust_sev[ cust_sev['billing_channel'] == 'OTT'].reset_index(drop=True)


# In[10]:


ott['num_trial_days'].unique()


# In[11]:


ott['renew'].unique()


# In[12]:


ott.columns


# In[13]:


ott.shape


# ### Preprocessing the data
# 

# In[14]:


# we delete those users whose cancel dates are ahead of their account creation date
pre_a = ott[ott['cancel_date'].isna() == 0]

pre_a = pre_a[ pre_a['cancel_date'] < pre_a['account_creation_date']]
pre_a[['subid','cancel_date','account_creation_date']].reset_index(drop=True).head()

error_id = list(pre_a['subid'])

ott = ott[~ott['subid'].isin(error_id)].reset_index(drop=True)


# In[15]:


ott['account_created_YM'] = ott['account_creation_date'].dt.strftime('%Y-%m')


# In[16]:


ott.head()


# ### Conducting the A/B testing

# In[17]:


ott.groupby('subid')[['last_payment']].count()['last_payment'].sum()


# In[18]:


a = ott[ott.payment_period == 0]
a.groupby('num_trial_days')['subid'].size()


# In[19]:


a.groupby('num_trial_days')['renew'].sum()


# In[20]:


ott_calc = ott[ott['payment_period'] == 0].reset_index(drop=True)
#ott_calc = ott_calc[ott_calc['trial_completed_TF'] != False].reset_index(drop=True)


# #### ott_test_a is for the purpose of spotting another data discrepancies about num_trial_day and trial_day_complted_TF and renew

# In[30]:


ott_test_a.columns


# In[37]:


ott_test_a = ott_calc[np.logical_and( ott_calc['renew'] == True, ott_calc['num_trial_days'] == 0) ]
ott_test_a[ott_test_a['trial_completed_TF'] == True][['subid','current_sub_TF', 'cancel_date', 'num_trial_days', 'trial_completed_TF', 'payment_period',
       'last_payment', 'next_payment', 'renew']]


# In[34]:


ott_test_a[ott_test_a['trial_completed_TF'] == False][['subid','current_sub_TF', 'cancel_date', 'num_trial_days', 'trial_completed_TF', 'payment_period',
       'last_payment', 'next_payment', 'renew']]


# In[209]:


ott_sum = ott_calc.groupby(['account_created_YM', 'num_trial_days'])[['subid','renew']].agg({'subid':'count','renew':'sum'}).rename(columns={'subid': 'num_exposures', 'renew': 'num_actual_trial'})


# In[210]:


ott_sum


# In[1]:


ott[np.logical_and( ott['renew'] == True, ott['trial_completed_TF'] == False) ]


# In[211]:


ott_sum['conversion_rate'] = ott_sum['num_actual_trial'] / ott_sum['num_exposures']


# In[212]:


ott_sum_edited = ott_sum.reset_index()


# In[213]:


ott_sum_edited['account_created_YM'].unique()


# In[214]:


ott_sum_edited.head()


# In[215]:


ott_sum_edited.columns = ['account_created_YM', 'trial_days', 'exposured_users',
       'converted_users', 'conversion_rate']


# In[216]:


pt_3 = ott_sum_edited[ott_sum_edited['trial_days'] == 14].pivot(index='account_created_YM', columns='trial_days', values=['exposured_users','converted_users'] )


# In[217]:


plot_3 = pt_3.plot(kind='bar', figsize=(10,8))

plot_3.set_xlabel("Month", rotation=0)
plot_3.set_ylabel('number of users')


# In[218]:


pt_2 = ott_sum_edited[ott_sum_edited['trial_days'] == 0].pivot(index='account_created_YM', columns='trial_days', values=['exposured_users','converted_users'] )


# In[219]:


plot_4 = pt_2.plot(kind='bar', figsize=(10,8))
plot_4.set_xlabel("Month")
plot_4.set_ylabel('number of users')


# In[201]:


xiagaogao = cust_sev[cust_sev['trial_completed_TF'] == False]


# In[206]:


xiagaogao[ xiagaogao['renew'] == True]['num_trial_days'].unique()


# In[220]:


ott_sum_edited


# In[61]:


ott_sum_edited.columns


# In[68]:


pivot_table = ott_sum_edited.pivot(index='account_created_YM', columns='trial_days', values='conversion_rate')


# In[79]:


pivot_table_2  = ott_sum_edited.pivot(index='account_created_YM', columns='trial_days', values=['conversion_rate','exposured_users'] )


# In[65]:


plot_1 = pivot_table.plot( figsize=(10,8), legend=True, 
                            title = "Conversion rates on both variants from 2019.6 till 2020.3")

plot_1.set_xticklabels(pivot_table.index, rotation = 45)
plot_1.set_xlabel("Month")
plot_1.set_ylabel('Conversion_rate')


# In[58]:


pivot_table


# In[81]:


pivot_table_2


# ### AB Testing set-up
# 
# $p_{14}$: conversion rate of 14 trial day for the population \
# $p_{0}$: conversion rate of 0 trial day for samples
# 
# <b>Null Hopothesis</b> \
# $H_{0}$: $p_{0}$ = $p_{14}$
# 
# <b>Alternative Hopothesis</b> \
# $H_{1}$: $p_{0}$ ≠ $p_{14}$
# 
# <b>Alpha Statement</b> \
# α = 0.05
# 
# <b>Critical Value</b> \
# $Z_{0.025}$ = 1.96
# $Z_{-0.025}$ = -1.96
# 
# <b>Decision Statement</b> \
# if Z statistic > 1.96 or < - 1.96, \
# we reject the Null Hypothesis \
# Else, we will not reject the Null Hypothesis, \
# but accept the Alternative Hypothesis
# 
# <b>Z statistics: One sample statistical test</b> 
# 

# In[68]:


pivot_table_2['z_statistics'] = ( pivot_table_2['conversion_rate'][0] - pivot_table_2['conversion_rate'][14]) / ( np.sqrt( pivot_table_2['conversion_rate'][14] * (1 - pivot_table_2['conversion_rate'][14] ) / pivot_table_2['exposured_users'][0] ) ) 


# In[69]:


pivot_table_2


# In[46]:


alpha = 0.05
z_ = stats.norm.ppf(1 - alpha / 2)
z = stats.norm.ppf(alpha / 2)


# From a monthly basis, all z statistics show that we should accept the alternative hypothesis the conversion rate of 0 trial day is better than that of 14 trial days

# #### Relation between revenue and channel

# <b>0_num_trial_days</b>

# In[135]:


ott_0 = ott[ ott['num_trial_days'] == 0 ]


# In[136]:


ott_0.columns


# In[137]:


ott_0[ott_0['subid'] == 20002686 ]


# In[138]:


ott_0['revenue_net_1month'].unique()


# In[156]:


sns.distplot(trial['revenue_net_1month'], hist=True, rug=False)


# In[152]:


trial = ott_0.groupby('subid')[['revenue_net_1month']].mean()


# In[140]:


ott_last = ott_0.groupby('subid')[['revenue_net_1month', 'payment_period', 'renew']].last().reset_index()
ott_last


# In[141]:


ott_last = ott_0.groupby('subid')[['revenue_net_1month', 'payment_period', 'renew']].last().reset_index()
ott_last = ott_last[ott_last['revenue_net_1month'] != 0].reset_index(drop=True)


# In[158]:


ott_last


# In[142]:


ott_last.payment_period.unique()


# In[143]:


ott_last = ott_last[ ott_last['payment_period'] <= 4 ].reset_index(drop=True)


# In[144]:


ott_last['payment_period'].unique()


# In[145]:


def cacl_ttl_rev(rev, period):
    if period == 4:
        return(rev * period)
    else:
        return( rev * (period + 1) )
    
ott_last['total_revenue'] = ott_last.apply(lambda row: cacl_ttl_rev( row['revenue_net_1month'], row['payment_period'] ), axis = 1)


# In[157]:


sns.distplot(ott_last['total_revenue'], hist=True, rug=False);


# In[148]:


ott_last.total_revenue.unique()


# In[80]:


ott[ ott['subid'] == 20002686]


# ## Google Channel

# In[82]:


ggle = cust_sev[ cust_sev['billing_channel'] == 'google'].reset_index(drop=True)
ggle.head()


# In[81]:


ggle['num_trial_days'].unique()


# ## iTunes Channel

# ### Preprocessing the data
# 

# In[39]:


it = cust_sev[ cust_sev['billing_channel'] == 'itunes'].reset_index(drop=True)
it.head()


# In[46]:


it['account_created_year'] = pd.DatetimeIndex(it['account_creation_date']).year
it['account_created_month'] = pd.DatetimeIndex(it['account_creation_date']).month


# In[47]:


it.renew.unique()


# In[48]:


it['num_trial_days'].unique()


# In[49]:


a = it[it.payment_period == 0]
a.groupby('num_trial_days')['subid'].size()


# In[50]:


a.groupby('num_trial_days')['renew'].sum()


# In[52]:


it_calc = it[it['payment_period'] == 0].reset_index(drop=True)


# In[54]:


it_sum = it_calc.groupby(['account_created_year', 'account_created_month', 'num_trial_days'])[['subid','renew']].agg({'subid':'count','renew':'sum'}).rename(columns={'subid': 'num_exposures', 'renew': 'num_actual_trial'})
it_sum


# In[39]:


cust_sev[cust_sev['subid']==27800927][['subid','num_trial_days','payment_period','billing_channel','revenue_net_1month','last_payment', 'next_payment', 'renew']]


# In[38]:


cust_sev[cust_sev['subid']==25914865][['subid','num_trial_days','payment_period','billing_channel','revenue_net_1month','last_payment', 'next_payment', 'renew']]


# In[ ]:





# In[70]:


subscribers[subscribers['subid'] == 25914865]


# In[23]:


cust_sev.reset_index().head(10)


# # Engagement

# datafile = open('engagement', 'rb')
# 
# eng = pickle.load(datafile)
# 
# datafile.close()

# In[214]:


eng.head(5)


# In[65]:


eng.columns


# In[16]:


eng[eng['app_opens'] > 100.0]


# In[67]:


eng[['app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']].max()


# ## descriptive analysis
# in <b>engagement</b> \
# there are 135019 unique users

# In[218]:


engagement.shape


# In[215]:


engagement['subid'].nunique()


# In[216]:


eng['year'] = pd.DatetimeIndex(eng['date']).year
eng['month'] = pd.DatetimeIndex(eng['date']).month


# In[217]:


eng['year'].unique()


# In[219]:


eng.drop_duplicates(subset='subid',keep='last',inplace=True)


# In[221]:


eng.groupby(['year','month'])[['date']].count()


# In[189]:


eng['month'].unique()


# In[25]:


eng_1.sort_values(['subid','date'], ascending = True)['subid'].nunique()


# In[ ]:


eng_1['subid'].nunique()


# In[30]:


list(engagement['payment_period'].unique())


# In[12]:


engagement.reset_index().head(5)


# # Subscribers
# 

# datafile = open('subscribers', 'rb')
# 
# sub = pickle.load(datafile)
# 
# datafile.close()

# In[4]:


sub['subid'].unique


# In[ ]:





# In[50]:


b.iloc[1:3].rename(index={680406: 'c'})


# In[ ]:


b = sub[ sub['attribution_survey'].isnull() == True ][['attribution_survey','attribution_technical']]


# In[39]:


bsub[ sub['attribution_survey'].isnull() == True ][['attribution_survey','attribution_technical']]


# In[42]:





# In[ ]:





# In[30]:


sub[['attribution_survey','attribution_technical']].iloc[1:3]


# In[24]:


a.rename(index={380374: 'a',496617:'b'})


# In[19]:


sub['attribution_survey'].unique()


# In[17]:


sub['attribution_technical'].unique()


# In[4]:





# In[9]:





# In[10]:


sub['country'].unique()


# In[83]:


sub.max()


# In[87]:


sub[np.logical_and( sub['age'] > 100, sub['age'] < 5000 )][['subid', 'age']].head()


# In[103]:


b = eng.drop_duplicates(subset='subid',keep='last',inplace=True)


# In[ ]:


eng.drop_duplicates(subset='subid',keep='last',inplace=True)


# In[107]:


sub.drop_duplicates(subset='subid',keep='first',inplace=True)


# In[111]:


b = eng[['subid']]


# In[112]:


pd.merge(sub,b,left_on='subid',right_on='subid',how='inner')


# ## Descriptive Analysis

# in <b>subscriber</b> \
# there are 227628 unique users

# In[170]:


subscribers.shape


# In[171]:


subscribers['subid'].nunique()


# In[14]:


subscribers.columns


# In[172]:


subscribers.head(5)


# In[198]:


sub['year'] = pd.DatetimeIndex(sub['account_creation_date']).year
sub['month'] = pd.DatetimeIndex(sub['account_creation_date']).month


# In[225]:


sub.groupby(['year','month'])[['subid']].count()


# In[16]:


subscribers[['country','plan_type']]


# In[15]:


subscribers['plan_type'].unique()


# # Ad spend
# 

# #### Preprocessing

# In[303]:


ad_spend['organic'] = ad_spend.shape[0] * [0]
ad_spend['other'] = ad_spend.shape[0] * [0]


# In[304]:


ad_spend.columns = ['facebook', 'email', 'search', 'bsi google', 'affiliate',
       'email_blast', 'pinterest', 'referral', 'organic', 'other']


# In[305]:


ad_spend


# In[15]:


subscribers.reset_index()


# In[41]:


cust_sev['subid'].nunique()


# In[42]:


cust_sev.columns = ['customer_service_rep_id', 'user_id', 'current_sub_TF', 'cancel_date',
       'account_creation_date', 'num_trial_days', 'trial_completed_TF',
       'billing_channel', 'revenue_net_1month', 'payment_period',
       'last_payment', 'next_payment', 'renew']


# In[ ]:


combined['']


# In[18]:


cust_sev[cust_sev['subid'] == 21724479]


# # M &  A

# ## Allocation and Attribution

# Due to the potential data inconsistency problem, we need to both take a look at crs and subscriber dataset the define a successful acquisition

# In[591]:


cust_sub = pd.merge(sub, cust_sev, left_on = 'subid', right_on = 'user_id', how = 'inner')


# In[577]:


cust_sub.shape


# In[578]:


cust_sub.head(5)


# In[579]:


cust_sub.columns


# ### Preprocessing

# In[592]:


# we delete the misatch cancel date and account creation date
cust_sub = cust_sub[~cust_sub['subid'].isin(error_id)].reset_index(drop=True)


# In[581]:


cust_sub.shape


# ##### Here we try to find out the mismatch between trial_completed

# In[288]:


cd_a = cust_sub[ cust_sub['cancel_date'] < cust_sub['trial_end_date'] ]
cd_a[['subid','plan_type','account_creation_date_x','trial_end_date','cancel_date','cancel_before_trial_end','initial_credit_card_declined','trial_completed_TF',
       'payment_period', 'last_payment', 'next_payment']]


# In[57]:


cd_b = cust_sub[ cust_sub['trial_completed_TF'] == False ]
cd_b


# In[58]:


# there are 72665 mismatch between trial_complted_TF and cancel before trial end
cd_b_id = set(cd_b['subid'])
cd_a_id = set(cd_a['subid'])


# In[59]:


cd_ab_id = cd_a_id.intersection(cd_b_id)
len(cd_ab_id)


# In[ ]:


cust_sev


# In[158]:


cd_a[~cd_a['subid'].isin(cd_ab_id)].reset_index(drop=True)[['subid','plan_type','account_creation_date_x','trial_end_date','cancel_date','cancel_before_trial_end','initial_credit_card_declined','trial_completed_TF',
       'payment_period', 'last_payment', 'next_payment']]


# In[99]:


cust_sub['trial_completed_TF'].isnull().sum()


# In[ ]:


cust_sev['trial_c']


# In[93]:


cust_sub[['cancel_before_trial_end','trial_end_date','cancel_date','account_creation_date_x','account_creation_date_y','trial_completed_TF']]


# ##### ['account_creation_date'] IN Subscribers IS THE SAME AS ['account_creation_date']  Customer_service_reps: 

# In[105]:


# THE SAME WITH THE TEST
cust_sub[ cust_sub['account_creation_date_x'] == cust_sub['account_creation_date_y'] ]


# In[317]:


cust_sub_aa = cust_sub[ np.logical_and(cust_sub['revenue_net'] != 0, cust_sub['renew'] == True) ]


# In[124]:


cust_sub_aa.reset_index(drop=True)['attribution_technical'].unique()


# ### Remapping the attribution channels

# In[318]:


cust_sub_aa['attribution_technical'].unique()


# In[319]:


ad_spend_channel = ['email', 'facebook', 'organic','email_blast', 'brand sem intent google', 'search','affiliate','referral','pinterest']


# In[320]:


def channel_remap(frame):
    global ad_spend_channel
    if frame in ad_spend_channel:
        if frame != 'brand sem intent google':
            return(frame)
        else:
            return('bsi google')
    elif '_organic' in frame:
        a = 'organic'
        return(a)
    else:
        return('other')


# In[321]:


cust_sub_aa['channel_edited'] = cust_sub_aa.loc[:,'attribution_technical'].apply(channel_remap)


# In[322]:


cust_sub_aa['channel_edited'].unique()


# In[294]:


cust_sub_aa.groupby('channel_edited')['subid'].size().to_frame().sort_values(by='subid', ascending=False).plot(kind='bar', figsize=(10,5), title = 'numbers of acquired users via 9 channels', legend=False )


# ### Average CAC

# In[295]:


a = cust_sub_aa.groupby('channel_edited')['subid'].size().to_frame()
a.sort_values(by='subid',ascending=False).T


# In[297]:


ad_spend


# In[299]:


ad_spend


# In[306]:


ad_spend.sum().to_frame().sort_values(by=0, ascending = False).plot(kind='bar', figsize =(9,5), title = 'total ad spend through 201906 til 202003 on 10 channels', legend=False)


# In[307]:


b = ad_spend.sum().to_frame()
b


# In[308]:


avg_cac = pd.merge(a,b,left_index=True, right_index=True, how='inner')


# In[309]:


avg_cac.columns = ['# of users','ad spend']
avg_cac['avg_cac'] = avg_cac['ad spend'] / avg_cac['# of users']


# In[310]:


avg_cac.sort_values(by='avg_cac', ascending = False)[['avg_cac']].plot(kind = 'bar', figsize =(9,6), title = 'average cac through each channel',color=[plt.cm.Paired(np.arange(len(avg_cac)))] )


# ### Marginal CAC

# In[323]:


ad_spend


# In[313]:


# cust_sub_aa = cust_sub_aa.drop(['account_creation_date_x'], axis=1)


# In[330]:


cust_sub_aa['account_creation_date_y'] = cust_sub_aa['account_creation_date_y'].dt.strftime('%Y-%m')


# In[327]:


cust_sub_aa['channel_edited'].unique()


# In[333]:


marg_cac = cust_sub_aa.groupby(['channel_edited','account_creation_date_y']).size().to_frame().reset_index()


# In[336]:


marg_cac['channel_edited'].unique()


# In[334]:


marg_cac.to_csv('mag_cac.csv')


# In[339]:


pd.read_csv('marginal_cac.csv', index_col=0)


# # Churn Modelling

# ### Engamenet Preprocessing

# In[582]:


eng = eng.reset_index(drop=True)


# In[583]:


eng.columns


# In[370]:


eng.isnull().sum()


# In[371]:


eng = eng.dropna().reset_index(drop=True)


# In[374]:


eng[ eng['subid'] == 20000062]


# In[387]:


eng.columns


# In[389]:


eng_pre = eng.groupby(['subid','payment_period'])[['app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']].sum().reset_index()


# In[390]:


eng_pre


# In[391]:


eng_pre.columns


# In[392]:


eng_pre_0 = eng_pre[ eng_pre['payment_period'] == 0]
eng_pre_0.columns = ['subid',
'payment_period',
'period_0_app_opens',
'period_0_cust_service_mssgs',
'period_0_num_videos_completed',
'period_0_num_videos_more_than_30_seconds',
'period_0_num_videos_rated',
'period_0_num_series_started']


# In[395]:


eng_pre_x0 = eng_pre[ eng_pre['payment_period'] != 0]


# In[399]:


eng_pre_x0.columns


# In[401]:


eng_pre_x0_drft = eng_pre_x0.groupby(['subid'])[['payment_period', 'app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']].agg({'payment_period':'count',
'app_opens':'sum',
'cust_service_mssgs':'sum',
'num_videos_completed':'sum',
'num_videos_more_than_30_seconds':'sum',
'num_videos_rated':'sum',
'num_series_started':'sum'})
eng_pre_x0_drft


# In[402]:


eng_pre_x0_drft.columns


# In[406]:


eng_pre_x0_drft['avg_app_opens'] = eng_pre_x0_drft['app_opens'] / eng_pre_x0_drft['payment_period']
eng_pre_x0_drft['avg_cust_service_mssgs'] = eng_pre_x0_drft['cust_service_mssgs'] / eng_pre_x0_drft['payment_period']
eng_pre_x0_drft['avg_num_videos_completed'] = eng_pre_x0_drft['num_videos_completed'] / eng_pre_x0_drft['payment_period']
eng_pre_x0_drft['avg_num_videos_more_than_30_seconds'] = eng_pre_x0_drft['num_videos_more_than_30_seconds'] / eng_pre_x0_drft['payment_period']
eng_pre_x0_drft['avg_num_videos_rated'] = eng_pre_x0_drft['num_videos_rated'] / eng_pre_x0_drft['payment_period']
eng_pre_x0_drft['avg_num_series_started'] = eng_pre_x0_drft['num_series_started'] / eng_pre_x0_drft['payment_period']


# In[408]:


eng_pre_x0_drft.columns


# In[437]:


eng_pre_0 = eng_pre_0.drop(['payment_period'], axis =1)
eng_pre_0


# In[410]:


eng_pre_x0_drft = eng_pre_x0_drft.drop(['payment_period', 'app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started'], axis=1)
eng_pre_x0_drft


# In[438]:


eng_all = pd.merge(eng_pre_0, eng_pre_x0_drft, right_on='subid',left_index=True, how='outer')


# In[439]:


eng_all.drop_duplicates(subset='subid',keep='first',inplace=True)


# In[440]:


eng_all = eng_all.reset_index(drop=True)


# In[570]:


eng_all = eng_all.fillna(0)


# In[584]:


eng_all


# ### cust_sub preprocessing

# In[593]:


cust_sub['age'].value_counts().to_frame().reset_index().sort_values(by='index', ascending=False)


# In[594]:


cust_sub['language'].unique()


# In[595]:


cust_sub = cust_sub[np.logical_and( cust_sub['age']  <= 100, cust_sub['age']  != 0) ]


# In[596]:


cust_sub.isnull().sum()


# In[597]:


#cust_sub['weekly_consumption_hour'] = 
cust_sub['weekly_consumption_hour'].fillna((cust_sub['weekly_consumption_hour'].mean()), inplace=True)


# In[599]:


cust_sub['weekly_consumption_hour'].isnull().sum()


# In[600]:


cust_sub.isnull().sum()


# In[507]:


#cust_sub = cust_sub.dropna(subset = ['package_type','num_weekly_services_utilized','preferred_genre','num_ideal_streaming_services','payment_type'])


# In[608]:


trial = pd.merge(eng_all, cust_sub, left_on='subid',right_on='subid',how='inner')
trial


# In[618]:


# quit_user = list(cust_sub [ cust_sub['cancel_date'] < cust_sub['trial_end_date'] ]['subid'])
quit_user = set(list(trial [ trial['cancel_date'] < trial['trial_end_date'] ]['subid']))


# In[619]:


trial[~trial['subid'].isin(quit_user)].reset_index(drop=True)


# In[629]:


trial


# In[607]:


#
cust_sub.drop_duplicates(subset='subid',keep='first', inplace=True)
cust_sub


# In[630]:


trial['how_long'] = (trial['next_payment'] - trial['last_payment']).dt.days
trial['how_long']


# In[631]:


trial.isnull().sum()


# In[572]:


for_churn = pd.merge(eng_all,cust_sub,left_on='subid',right_on='subid',how='inner')


# In[573]:


for_churn


# In[552]:


for_churn['intended_use'].unique()


# In[554]:


for_churn['current_sub_TF'].unique()


# In[632]:


def churn(frame):
    if frame == True:
        return(0)
    else:
        return(1)


# In[633]:


trial['churn_status'] = trial['current_sub_TF'].apply(churn)


# In[634]:


trial


# In[636]:


trial.isnull().sum()


# In[640]:


trial.columns


# In[644]:


X = trial[[ 'subid', 'period_0_app_opens', 'period_0_cust_service_mssgs',
       'period_0_num_videos_completed',
       'period_0_num_videos_more_than_30_seconds', 'period_0_num_videos_rated',
       'period_0_num_series_started', 'avg_app_opens',
       'avg_cust_service_mssgs', 'avg_num_videos_completed',
       'avg_num_videos_more_than_30_seconds', 'avg_num_videos_rated',
       'avg_num_series_started','intended_use', 'weekly_consumption_hour','num_trial_days','initial_credit_card_declined','churn_status']]


# In[651]:


X = X.dropna().reset_index(drop=True)


# In[652]:


X.isnull().sum()


# In[655]:


X = pd.get_dummies(X,columns=['intended_use','num_trial_days','initial_credit_card_declined'],drop_first=True)


# In[656]:


X


# In[657]:


# Data preprocessing module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

#np.set_printoptions(precision=2)

np.set_printoptions(precision=4)


# In[662]:


y = X[['churn_status']]
x = X.copy().drop(['churn_status','subid'], axis=1)


# In[664]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)

# scale
scaler = StandardScaler()

x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)


# In[665]:


# Logistic Regression: 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# instantiate the model ( using the default parameters)
logreg = LogisticRegression().fit(x_train_scale, y_train)
penalty = ['l1', 'l2']
C = np.logspace(-1, 4, 10) 
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(x, y)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[669]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
C_opt = 774.26 # the optimal parameter C
logreg = LogisticRegression(penalty='l2',C=C_opt).fit(x_train_scale, y_train)

print("Training accuracy",logreg.score(x_train_scale,y_train))
# use test data to see the prediction
y_pred = logreg.predict(x_test_scale).astype('int64')
result = pd.DataFrame()
y_pred_prob = logreg.predict_proba(x_test_scale)
result['log_01'] = y_pred
result['log_prob'] = y_pred_prob[:,1]
result.head(5)

y_test = y_test.astype('int64')
print("Test Accuracy",metrics.accuracy_score(y_test,y_pred))
print("Test Precision",metrics.precision_score(y_test,y_pred))
print("Test Recall",metrics.recall_score(y_test,y_pred))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[670]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[671]:


import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[676]:


from sklearn.ensemble import GradientBoostingClassifier
param_test = {'max_features':range(2, 10 ,1),
                'n_estimators': range(100, 150, 10),
                'max_depth':  range(2, 5, 1)
}

estimator = GradientBoostingClassifier()
gsearch = GridSearchCV(estimator , param_grid = param_test, cv=5)
gsearch.fit(x_train_scale, y_train)
gsearch.best_params_, gsearch.best_score_
print('best score is:',str(gsearch.best_score_))
print('best params are:',str(gsearch.best_params_))


# In[680]:


def plot_feature_importances(clf, feature_names):
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)

clf = GradientBoostingClassifier(max_depth = 2,max_features=9, n_estimators = 140)
clf_best = clf.fit(x_train_scale, y_train)
plt.figure(figsize=(14,10), dpi=80)
plot_feature_importances(clf_best, x_train.columns)
plt.show()

print('Feature importances: {}'.format(clf_best.feature_importances_))


# In[686]:


x_test


# In[688]:


hey = y_test.reset_index(drop=True)
hey


# In[689]:


result_1 = pd.merge(hey, result, left_index=True, right_index=True, how='inner')
result_1


# In[701]:


X_x = pd.merge(X, x_test, left_index=True,right_index=True, how='inner').reset_index(drop=True)
X_x


# In[702]:


finale = pd.merge(X_x, result_1, left_index=True, right_index=True)


# In[706]:


to_match = finale[['subid','log_prob']]


# In[707]:


clv_calc = pd.merge(cust_sub, to_match, left_on='subid',right_on='subid', how='inner')


# In[717]:


clv_fin = clv_calc[['subid','attribution_technical','monthly_price','account_creation_date_x','log_prob']]


# In[719]:


clv_fin['date_creation'] = clv_fin['account_creation_date_x'].dt.strftime('%Y-%m')


# In[720]:


clv_fin


# In[ ]:


def channel_remap(frame):
    global ad_spend_channel
    if frame in ad_spend_channel:
        if frame != 'brand sem intent google':
            return(frame)
        else:
            return('bsi google')
    elif '_organic' in frame:
        a = 'organic'
        return(a)
    else:
        return('other')


# In[721]:


clv_fin['channel'] = clv_fin['attribution_technical'].apply(channel_remap)


# In[746]:


clv_fin = clv_fin.drop(['attribution_technical','account_creation_date_x'], axis=1)
clv_fin


# In[724]:


en = pd.read_csv('marginal_cac.csv',index_col=0)


# In[726]:


en = en.reset_index()


# In[735]:


date = list(clv_fin['date_creation'].sort_values().unique())


# In[736]:


en['index'] = date


# In[747]:


to_calc = pd.merge(clv_fin, en, left_on='date_creation',right_on='index')
to_calc


# In[756]:


to_calc.loc[0,'email']


# In[757]:


cac = []
for i, q in to_calc.iterrows():
    a = str(q.channel)
    c = to_calc.loc[i, a]
    cac.append(c)


# In[760]:


to_calc['cac'] = cac


# In[765]:


to_calc_1 = to_calc[['subid','monthly_price','log_prob','channel','cac']]


# In[766]:


to_calc_1['clv'] = to_calc_1['monthly_price'] * 1.1/(0.1+to_calc_1['log_prob'])-to_calc_1['cac']


# In[771]:


to_calc_1.to_csv('clv.csv')


# In[695]:


sub['subid'].value_counts()

