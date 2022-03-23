project_dir= "E:/Python WD/ml_pipeline"
import os
os.chdir(project_dir)
import joblib
series = joblib.load('temporary_objects/XY')
Y = series['Y']




def recommend_balancing(Y):
    def class_distribution(Y):
        import pandas as pd
        result = pd.DataFrame(Y.value_counts().reset_index()).rename(columns={0: 'count'})
        result['percentage'] = round(result['count'] / result['count'].sum(), 2)
        result
        return result
    result= class_distribution(Y)
    n_classes= result.shape[0]
    expected_mean= 1/n_classes

    permissible_deviation= .2* expected_mean

    if (result.percentage[0] > expected_mean+permissible_deviation) or (result.percentage[0] < expected_mean-permissible_deviation):
        recommendation= True
    else:
        recommendataion= False
    return recommendation





