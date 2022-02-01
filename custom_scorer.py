
confusion_matrix_scorer(Y, estimator.predict(X))
cv_results['test_f1'].mean()
cv_results['train_f1'].mean()
cv_results['test_f1'].std()
cv_results['train_f1'].std()



def confusion_matrix_scorer(Y, Y_pred):
    #Y_pred = clf.predict(X)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y, Y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

def my_custom_loss_func(Y, Y_pred):
    import numpy as np
    diff = np.abs(Y - Y_pred).max()
    return np.log1p(diff)

