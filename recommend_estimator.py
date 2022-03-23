# project_dir= "E:\Python WD/ml_pipeline"
# import os
# os.chdir(project_dir)
# path= 'output/metric_df.csv'
# import pandas as pd
# metrics_df= pd.read_csv(path)
# metrics_to_consider= ['f1_score','auc']

def recommend_estimator(metrics_df, problem_type= 'classification'):
    import pandas as pd
    if problem_type== 'classification':
        metrics_to_consider = ['f1_score', 'auc']
    metrics_df= metrics_df.set_index('Metric')
    temp= metrics_df.loc[metrics_df.index.isin(metrics_to_consider),:]
    temp= temp.astype(float) ## ensure that the data type is float
    result= pd.DataFrame(temp.mean(), columns=['aggregate_metric'])
    result['rank']= result['aggregate_metric'].rank(ascending=False)
    recommended_estimator= result.loc[result['rank']==1.0,:].index[0]
    return recommended_estimator

