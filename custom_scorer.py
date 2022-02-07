
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




from sklearn.metrics import confusion_matrix, make_scorer
confusion_matrix_scorer = make_scorer(confusion_matrix)


metrics['auc'] = roc_auc_score(Y, Y_PRED)
###custom metric loss_index

###precision recall curve
precisions, recalls, thresholds = precision_recall_curve(Y, Y_PRED_prob)

# metrics['prec_recall_plot'].show()
metrics['gini'] = 2 * roc_auc_score(Y, Y_PRED_prob) - 1
# etrics['confusion_matrix']= confusion_matrix(Y, Y_PRED,labels=[0,1])
positives, negatives = confusion_matrix(Y, Y, labels=[0, 1])
TP, FP = positives[0], positives[1]
FN, TN = negatives[0], negatives[1]

metrics['TP'] = TP
metrics['FP'] = FP
metrics['FN'] = FN
metrics['TN'] = TN
