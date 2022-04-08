# import pandas as pd
# import seaborn as sns
# data= sns.load_dataset('iris')
# col= 'sepal_length'

def duplicate_column(data, col, suffix='_dup'):
    data[col+suffix]= data[col]
    return data



def IQR_graph_data(data, col):

    ### return a list containig
    result_dict= {'data':[data[col].quantile(0),data[col].quantile(.25),data[col].quantile(.5),data[col].quantile(.75),data[col].quantile(1)]}
    result_dict['name']= 'Observations'
    return result_dict

# Outlier - Percentile find outliers and graph data API

def find_outliers(data, col):
    result_dict = {}
    Q1 = data[col].quantile(.25)
    Q3 = data[col].quantile(.75)
    range = data[col].quantile(.75) - data[col].quantile(.25)

    obs_list= [data[col].quantile(0),data[col].quantile(.25),data[col].quantile(.5),data[col].quantile(.75),data[col].quantile(1)]
    observations_dict= {'name': 'Observations', 'data': obs_list}

    upper_threshold= Q3+1.5* range
    lower_threshold= Q1-1.5* range

    upper_outlier_list= data[col][data[col]> upper_threshold].to_list()
    lower_outlier_list= data[col][data[col]< lower_threshold].to_list()
    outlier_list= upper_outlier_list+ lower_outlier_list
    outlier_list_result = []
    for i in outlier_list:
        outlier_list_result.append([0, i])
    outlier_list_result

    outlier_dict= {'name': 'Outlier', 'data': outlier_list_result}
    result_dict= {'observations_dict':observations_dict,'outlier_dict': outlier_dict }
    return result_dict



def outlier_graph_data_iqr(data, col):
    result_dict = {}
    Q1 = data[col].quantile(.25)
    Q3 = data[col].quantile(.75)
    range = data[col].quantile(.75) - data[col].quantile(.25)

    obs_list= [data[col].quantile(0),data[col].quantile(.25),data[col].quantile(.5),data[col].quantile(.75),data[col].quantile(1)]
    observations_dict= {'name': 'Observations', 'data': obs_list}

    upper_threshold= Q3+1.5* range
    lower_threshold= Q1-1.5* range

    upper_outlier_list= data[col][data[col]> upper_threshold].to_list()
    lower_outlier_list= data[col][data[col]< lower_threshold].to_list()
    outlier_list= upper_outlier_list+ lower_outlier_list
    outlier_list_result = []
    for i in outlier_list:
        outlier_list_result.append([0, i])
    outlier_list_result

    outlier_dict= {'name': 'Outlier', 'data': outlier_list_result}
    result_dict= {'observations_dict':observations_dict,'outlier_dict': outlier_dict }
    return result_dict

# lower_bound_percentile= .01
# upper_bound_percentile= .99
# import joblib
# series = joblib.load('temporary_objects/XY')
# X, Y = series['X'], series['Y']
# data= X
# col= 'BloodPressure'



def outlier_graph_data_percentile(data, col, lower_bound_percentile, upper_bound_percentile):
    result_dict = {}
    lower_bound = data[col].quantile(lower_bound_percentile)
    upper_bound = data[col].quantile(upper_bound_percentile)

    obs_list= [lower_bound, data[col].quantile(.5), upper_bound]
    observations_dict= {'name': 'Observations', 'data': obs_list}

    upper_threshold= upper_bound
    lower_threshold= lower_bound

    upper_outlier_list= data[col][data[col]> upper_threshold].to_list()
    lower_outlier_list= data[col][data[col]< lower_threshold].to_list()
    outlier_list= upper_outlier_list+ lower_outlier_list
    outlier_list_result = []
    for i in outlier_list:
        outlier_list_result.append([0, i])
    outlier_list_result

    outlier_dict= {'name': 'Outlier', 'data': outlier_list_result}
    result_dict= {'observations_dict': observations_dict,'outlier_dict': outlier_dict }
    return result_dict


## how to access observations
#result_dict['observations_dict']['data']

## how to access outliers
#result_dict['outlier_dict']['data']
#
# result_dict.keys()
# result_dict= outlier_graph_data_percentile(data, col, lower_bound_percentile, upper_bound_percentile)










