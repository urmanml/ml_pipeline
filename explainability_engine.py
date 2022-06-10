

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import shap

def generate_explanability(estimator, dataset):
    """
    Can be triggered alongside when prdiction api is triggered. Will generate teh explainability dataframe. We need to use separate api for generating visualizations
    :param estimator: trained estimator on which explainability is to be generated
    :param dataset: Dataset for which explainability is to be generated
    :return: shap_values_dataset
    """
    import shap
    est_name= str(estimator.__class__.__name__)

    if est_name == 'XGBClassifier':
        explainer = shap.TreeExplainer(estimator)
        shap_values_dataset = pd.DataFrame(explainer.shap_values(dataset), columns=dataset.columns,
                                           index=dataset.index)


    elif est_name == 'DecisionTreeClassifier':
        explainer = shap.TreeExplainer(estimator)
        shap_values_dataset = pd.DataFrame(explainer.shap_values(dataset)[1], columns=dataset.columns,
                                           index=dataset.index)

    else:# est_name== 'LogisticRegression':for all other estimators

        f = lambda x: estimator.predict_proba(x)[:, 1]
        # shap_values= shap.KernelExplainer(estimator.predict, dataset)
        # explainer= shap.KernelExplainer(f,dataset_row)
        explainer = shap.KernelExplainer(f, shap.sample(X))
        import shap
        f = lambda x: estimator.predict_proba(x)[:, 1]
        # shap_values= shap.KernelExplainer(estimator.predict, dataset)
        explainer = shap.KernelExplainer(f, shap.sample(dataset))
        shap_values_dataset = pd.DataFrame(explainer.shap_values(dataset), columns=dataset.columns,
                                           index=dataset.index)

    #shap_values_dataset = pd.DataFrame(explainer.shap_values(dataset)[0], columns=dataset.columns, index=dataset.index)
    return shap_values_dataset

    shap_values_dataset= generate_explanability(estimator, dataset)

    print(shap_values_dataset.head())
#shap_values_dataset= generate_explanability(estimator, dataset)

def generate_group_plot_data(shap_values_dataset):
    plot_data= shap_values_dataset.abs().sum().sort_values(ascending=False).reset_index(name='value')
    plot_data.columns= ['feature', 'feature_contribution']
    plot_data['feature_contribution']= plot_data['feature_contribution'] / plot_data['feature_contribution'].sum() * 100
    return plot_data
#plot_data= generate_group_plot_data(shap_values_dataset)
#sns.barplot(x= plot_data.feature_contribution, y= plot_data.feature)


def generate_entity_plot_data(shap_values_dataset_row, dataset_row):

    shap_values_dataset_row.T
    plot_data = shap_values_dataset_row.T.reset_index()
    plot_data.columns = ['feature', "feature_contribution"]
    plot_data['feature_contribution_abs']= plot_data['feature_contribution'].abs()
    plot_data['feature_value']= dataset_row.iloc[0,:].values
    plot_data = plot_data.sort_values(by='feature_contribution_abs', ascending=False)
    plot_data['feature']= plot_data['feature']+' = '+ plot_data['feature_value'].astype(str)

    plot_data= plot_data.drop(columns= ['feature_contribution_abs', 'feature_value'])
    return plot_data

##sample input filter out a row of data to generate visualize plots for a row
#shap_values_dataset_row= shap_values_dataset.iloc[1:2,:]
#dataset_row= dataset.iloc[1:2,:]

### generate plot data
#plot_data= generate_entity_plot_data(shap_values_dataset_row, dataset_row)

## generate sample plot
#import waterfall_chart
#waterfall_chart.plot(plot_data.feature, plot_data.feature_contribution)


def generate_feature_plot_data(shap_values_dataset, dataset, feature):
    plot_data= pd.DataFrame({'feature_value':dataset[feature], 'feature_contribution': shap_values_dataset[feature]}).reset_index(drop=True)
    return plot_data
# feature= "Glucose"
# plot_data= generate_feature_plot_data(shap_values_dataset, dataset, feature)
# #plot_data= plot_data.sample(10)
# sns.regplot(x= plot_data.feature_value, y= plot_data.feature_contribution)
# plt.title(feature)
