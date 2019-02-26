import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

# Print null data
def print_null(df):
    for col in df:
        if df[col].isnull().any():
            print('%s has %.0f null values: %.3f%%'%(col, df[col].isnull().sum(), df[col].isnull().sum()/df[col].count()*100))
            
# Impute null data using random sampling
def impute_na(X_train, df, variable):
    # make temporary df copy
    temp = df.copy()
    
    # extract random from train set to fill the na
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0, replace=True)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]

# Extract money value
def extract_money(text):
    if isinstance(text, str):
        text = text.replace('$','')
        text = text.replace(',','')
        money = float(text.replace('$',''))
    else:
        money=text
    return money

# Clipping outliers
def clipping_outliers(X_train, df, var):
    IQR = X_train[var].quantile(0.75)-X_train[var].quantile(0.25)
    lower_bound = X_train[var].quantile(0.25) - 4*IQR
    upper_bound = X_train[var].quantile(0.75) + 4*IQR
    no_outliers = len(df[df[var]>upper_bound]) + len(df[df[var]<lower_bound])
    print('There are %i outliers in %s: %.3f%%' %(no_outliers, var, no_outliers/len(df)))
    df[var] = df[var].clip(lower_bound, upper_bound)
    return df

# Plot overview
def plot_overview(df, column, top_count=5):
    agg_func = {'id':['count'],
                   'booking_rate(%)':['mean'],
                   'price_per_person':['mean'],
                   'daily_revenue' : ['mean']} #'revenue_per_guest':['mean']
    temp_df = df.groupby(column).agg(agg_func)
    temp_df.columns = ['_'.join(col)for col in temp_df.columns.values]
    temp_df = temp_df.sort_values(by='id_count', ascending=False)
    temp_df.reset_index(inplace=True)
    if len(temp_df)>top_count:
        temp_df = temp_df.loc[:top_count-1,:]
        
    temp_df = temp_df.sort_values(by='daily_revenue_mean', ascending=False)

    # Plot count and price
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(211)
    ax12 = ax1.twinx()
    temp_df.plot(x=column, y='id_count', kind='bar', color='blue', ax=ax1, width=0.4, position=1, legend=False)
    temp_df.plot(x=column, y='price_per_person_mean', kind='bar', color='red',ax=ax12, width=0.4, position=0, legend=False)
    #temp_df['id_count'].plot(kind='bar', color='blue', ax=ax1, width=0.4, position=1)
    #temp_df['price_mean'].plot(kind='bar', color='red',ax=ax12, width=0.4, position=0)
    ax1.set_ylabel('Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelbottom=False)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax12.set_ylabel('Price per person ($)', color='red')
    ax12.tick_params(axis='y', labelcolor='red', labelbottom=False)
    ax12.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xlim(left=-.6)

    ax2 = fig1.add_subplot(212)
    ax22 = ax2.twinx()
    temp_df.plot(x=column, y='booking_rate(%)_mean', kind='bar', color='blue', ax=ax2, width=0.4, position=1, legend=False)
    #temp_df.plot(x=column, y='revenue_per_guest_mean', kind='bar', color='red',ax=ax22, width=0.25, position=0)
    temp_df.plot(x=column, y='daily_revenue_mean', kind='bar', color='red',ax=ax22, width=0.4, position=0, legend=False)
    ax2.set_ylabel('Booking rate(%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue', labelbottom=False)
    ax2.tick_params(axis='x', rotation=30)
    ax22.set_ylabel('Daily revenue ($)', color='red')
    ax22.tick_params(axis='y', labelcolor='red', labelbottom=False)
    #ax22.tick_params(axis='x', rotation=45)
    ax2.set_xlabel(column)
    ax2.set_xlim(left=-.6)
    ax1.set_title('%s statistics' %column)
    plt.tight_layout()