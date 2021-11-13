# Date Disimilar
import pandas as pd, numpy as np
######sampple inputs
#prepare dataset
import seaborn as sns
from datetime import datetime
data= sns.load_dataset('iris')
data['col_date']= '2021-11-07'
data['col_date']= pd.to_datetime(data['col_date'])
test_data= data
test_data.loc[0,'sepal_width']= 'test_data'
test_data.loc[0,'col_date']= 'test_data'
data.info()

desired_datatype_dict= {'sepal_length': 'float',
 'sepal_width': 'float',
 'petal_length': 'float',
 'petal_width': 'float',
 'species': 'str',
 'col_date':'datetime'}


##--------------------------------------------------------------------------------
def alldtype_mismatch_preview_col(col, data,  desired_datatype_dict):
    from itertools import repeat
    invalid_mask= map(isinstance, data[col], repeat(eval(desired_datatype_dict[col])))
    invalid_mask=[not e for e in invalid_mask]
    per_invalid = round(sum(invalid_mask)/len(invalid_mask)*100,2)
    count_invalid = sum(invalid_mask)
    return per_invalid

##sample execution
col= 'col_date'
alldtype_mismatch_preview_col(col, data,  desired_datatype_dict)



def alldtype_mismatch_preview_df(data, desired_datatype_dict):
    from itertools import repeat
    col_list = data.columns.tolist()
    ##wrapper function applies to a single column
    results= map(alldtype_mismatch_preview_col, col_list, repeat(data), repeat(desired_datatype_dict))
    result_list= [e for e in results]

    result_dict= dict(pd.DataFrame({'column': desired_datatype_dict.keys(), 'invalid_percentage': result_list}).values)
    return result_dict

##sample execution
alldtype_mismatch_preview_df(data, desired_datatype_dict)



##--------------------------------------------------------
def alldtype_mismatch_remove(col, data,  desired_datatype_dict):
    from itertools import repeat
    invalid_mask= map(isinstance, data[col], repeat(eval(desired_datatype_dict[col])))
    invalid_mask=[not e for e in invalid_mask]
    valid_mask= np.logical_not(invalid_mask)
    return data.loc[valid_mask,:]

##sample execution
alldtype_mismatch_remove(col, data,  desired_datatype_dict)

def alldtype_mismatch_replace(col, data,  desired_datatype_dict, replace_value= 'test_data_replace'):
    data= data.copy()
    from itertools import repeat
    invalid_mask= map(isinstance, data[col], repeat(eval(desired_datatype_dict[col])))
    invalid_mask=[not e for e in invalid_mask]
    valid_mask= np.logical_not(invalid_mask)
    data.loc[invalid_mask, col]= replace_value
    return data

##sample execution
alldtype_mismatch_replace(col, data,  desired_datatype_dict, replace_value= 'test_data_replace')


