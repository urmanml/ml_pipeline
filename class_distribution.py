def class_distribution(Y):
    import pandas as pd
    result= pd.DataFrame(Y.value_counts().reset_index()).rename(columns={0:'count'})
    result['percentage']= round(result['count']/result['count'].sum(),2)
    result
    return result
