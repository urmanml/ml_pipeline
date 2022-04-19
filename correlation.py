# import joblib
# series = joblib.load('temporary_objects/XY')
# X, Y = series['X'], series['Y']
#
# dataset= X
# X.select_dtypes(include='number')


def correlation(dataset, method= None, columns= None):

    ## subset dataframe by user provided set of columns

    if (columns != None) and (columns!=[]):
    ## filter only numeric columns
        dataset= dataset[columns]
        dataset= dataset.select_dtypes(include='number')
    else:
        dataset= dataset.select_dtypes(include='number')



    methods= ['pearson','kendall','spearman']
    if method== 'spearman':
        result= dataset.corr( method= method)
    if method== 'kendall':
        result= dataset.corr( method= method)
    else:
        result= dataset.corr( method= 'pearson')

    result_dict= dict(result)
    return result_dict
# correlation(dataset,method='spearman',columns= ['Pregnancies', 'Age'])
# correlation(dataset,method='kendall',columns= ['Pregnancies', 'Age'])
# result= correlation(dataset,method='pearson',columns= ['Pregnancies', 'Age'])



