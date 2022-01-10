#X, Y = series['X'], series['Y']
def class_imbalance(X, Y):
    import numpy as np
    from sklearn.utils import resample
    #
    # Create oversampled training data set for minority class
    #
    temp= Y.value_counts()
    majority_class= temp.idxmax()[0]
    minority_class= temp.idxmin()[0]
    X_imbalanced= X.reset_index(drop=True)
    Y_imbalanced= Y.reset_index(drop=True)
    X_oversampled, Y_oversampled = resample(X_imbalanced[(Y_imbalanced.values == minority_class)],
                                            Y_imbalanced[Y_imbalanced.values == minority_class],
                                            replace=True,
                                            n_samples=X_imbalanced[Y_imbalanced.values == majority_class].shape[0],
                                            random_state=123)
    # Append the oversampled minority class to training data and related labels
    #
    import pandas as pd
    X_balanced = pd.concat([X[Y.values == majority_class], X_oversampled], axis= 0)
    Y_balanced = pd.concat( [Y[Y.values == majority_class], Y_oversampled], axis= 0)

    return X_balanced, Y_balanced

#temp1, temp2= class_imbalance(X, Y)

