# Date Disimilar
import pandas as pd, numpy as np

#prepare dataset
import seaborn as sns
data= sns.load_dataset('iris')
data['col_date']= '2021-11-07'
data['col_date']= pd.to_datetime(data['col_date'])
test_data= data
test_data.loc[0,'sepal_width']= 'test_data'
test_data.loc[0,'col_date']= 'test_data'


### generic mismatch function
def alldtype_mismatch(col, data,  desired_datatype_dict):
    cnt=0
    c1=data[col].isna().sum()

    for i in data[col]:
        if isinstance(i, eval(desired_datatype_dict[col])):
            cnt+=1
    z = len(data[col]) - cnt - c1
    per=(z/len(data[col]))*100
    return per

### function to call
#alldtype_mismatch(data, col, desired_datatype_dict[col])




def dtype_mismatch2(data,desired_datatype_dict):
    from itertools import repeat
    col_list = data.columns.tolist()
    ##wrapper function applies to a single column
    results= map(alldtype_mismatch, col_list, repeat(data), repeat(desired_datatype_dict))
    result_list=[]

    for result in results:
        result_list.append(result)

    result_dict= dict(pd.DataFrame({'column': desired_datatype_dict.keys(), 'invalid_percentage': result_list}).values)
    return result_dict


def dtype_mismatch3(data, desired_datatype_dict):
    def alldtype_mismatch(col, data, desired_datatype_dict):
        cnt = 0
        c1 = data[col].isna().sum()

        for i in data[col]:
            if isinstance(i, eval(desired_datatype_dict[col])):
                cnt += 1
        z = len(data[col]) - cnt - c1
        per = (z / len(data[col])) * 100
        return per


    from itertools import repeat
    col_list = data.columns.tolist()
    ##wrapper function applies to a single column
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results= executor.map(alldtype_mismatch, col_list, repeat(data), repeat(desired_datatype_dict))
    result_list=[]

    for result in results:
        result_list.append(result)

    result_dict= dict(pd.DataFrame({'column': desired_datatype_dict.keys(), 'invalid_percentage': result_list}).values)
    return result_dict

##calling this function
#dtype_mismatch2(data, desired_datatype_dict)

#data.drop(columns=['Unnamed: 0']).to_csv('E:/Python WD/ml_pipeline/input/sample_data.csv')
