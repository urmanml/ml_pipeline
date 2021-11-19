import pandas as pd
import seaborn as sns
data= sns.load_dataset('iris')
col= 'sepal_length'

def duplicate_column(data, col, suffix='_dup'):
    data[col+suffix]= data[col]
    return data



def IQR_graph_data(data, col):
    result_dict={}
    result_dict['min']= data[col].quantile(0)
    result_dict['Q1']= data[col].quantile(.25)
    result_dict['median']= data[col].quantile(.5)
    result_dict['Q3']= data[col].quantile(.75)
    result_dict['max']= data[col].quantile(1)
    result_dict['range']=  result_dict['Q3'] - result_dict['Q1']
    return result_dict

# Outlier - Percentile find outliers and graph data API

def find_outliers(data, col):
    result_dict = {}
    result_dict['min'] = data[col].quantile(0)
    result_dict['Q1'] = data[col].quantile(.25)
    result_dict['median'] = data[col].quantile(.5)
    result_dict['Q3'] = data[col].quantile(.75)
    result_dict['max'] = data[col].quantile(1)
    result_dict['range'] = result_dict['Q3'] - result_dict['Q1']
    upper_threshold= result_dict['Q3']+1.5* result_dict['range']
    lower_threshold= result_dict['Q1']-1.5* result_dict['range']

    result_dict['upper_outlier_list']= data[col][data[col]> upper_threshold].to_list()
    result_dict['lower_outlier_list']= data[col][data[col]< lower_threshold].to_list()
    return result_dict



