https://medium.com/ing-blog/model-explainability-how-to-choose-the-right-tool-6c5eabd1a46a


import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

