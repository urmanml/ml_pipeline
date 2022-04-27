## Testing data
# import joblib
# project_dir= "E:/Python WD/ml_pipeline"
# import os
# os.chdir(project_dir)
# series = joblib.load('temporary_objects/XY')
# X, Y = series['X'], series['Y']
# dataset= X
# import pandas as pd
# dataset['Age_band']= pd.cut(dataset['Age'],bins= [0,25,45,100]).astype('object')
#
# dataset.reset_index(drop=True, inplace= True)
# dataset.loc[0:100,'bool_column']= True
# dataset.loc[100:,'bool_column']= False

# data= dataset
# col= 'bool_column'
# bar_chart(col, data)

####
def pie_chart(col, data):
    series= data[col]
    series= series.astype('str')
    proportions= series.value_counts()/series.value_counts().sum()*100
    proportions= {'label': proportions.index.values.tolist(),'proportions':proportions.values.tolist()}
    import json
    result= json.dumps(proportions)
    return result



def bar_chart(col, data):
    series= data[col]
    series= series.astype('str')
    counts= series.value_counts()
    counts= {'label': counts.index.values.tolist(),'counts':counts.values.tolist()}
    import json
    result= json.dumps(counts)
    return result


def boxplot_iqr(col, data):
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


def eda_master(data):
    """
    structure:
    -numeric/categorical ( for each column of type numeric, use box plot graph, for each column of type categorical, use pie chart )
        -columns (names of columns)
        -data ( graph data for each of the column)

    :param data:
    :return:
    """
    ## identify numeric and
    data_numeric= data.select_dtypes(include='number')
    cols= data_numeric.columns.to_list()
    f= lambda col: boxplot_iqr(col, data_numeric)

    # map the function to the list and pass
    # function and list_ranges as arguments
    #pool.map(np.exp, [1,2])
    result= list(map(f, cols))
    dict_result_numeric= {'columns': cols, 'data': result}

    ### string types
    data_string= data.select_dtypes(include=['category', 'object'])
    cols= data_string.columns.to_list()
    f= lambda col: pie_chart(col, data_string)

    # map the function to the list and pass
    # function and list_ranges as arguments
    #pool.map(np.exp, [1,2])
    result= list(map(f, cols))
    dict_result_string= {'columns': cols, 'data': result}

    ## identify bool types
    data_bool= data.select_dtypes(include='bool')
    cols= data_bool.columns.to_list()
    f= lambda col: bar_chart(col, data_bool)

    # map the function to the list and pass
    # function and list_ranges as arguments
    #pool.map(np.exp, [1,2])
    result= list(map(f, cols))
    dict_result_bool= {'columns': cols, 'data': result}


    result= {'numeric': dict_result_numeric, 'string': dict_result_string, 'bool': dict_result_bool}


    return result

## sample execution
#eda_master(data)


### how to access
# result["numeric"]['columns'] ### gives list of column names
# result["numeric"]['data'] ### gives list of graph data
# result["string"]['data']


