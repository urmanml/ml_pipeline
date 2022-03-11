import joblib
series = joblib.load('temporary_objects/XY')
X, Y, test_X, test_Y = series['X'], series['Y'], series['test_X'], series['test_Y']




def nonlinear(X, n_components):
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=n_components, init='random').fit_transform(X)
    return X_embedded


def linear(X, n_components):
    from sklearn.decomposition import PCA
    X_embedded = PCA(n_components=n_components, init='random').fit_transform(X)

    return X_embedded

def dimensionality_reduction(X, n_components, approach):
    if approach== 'linear':
        X_embedded= linear(X, n_components)

    elif approach== 'nonlinear':
        X_embedded= nonlinear(X, n_components)
    return X_embedded

X_embedded= dimensionality_reduction(X, 3, 'nonlinear')

data= Y
data[['pc1','pc2']]= X_embedded
sns.scatterplot(x= X_embedded[:,0], y= X_embedded[:,1])

import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))
plt.show()

sns.scatterplot(
    x="pc1", y="pc2",
    hue="Outcome",
    palette=sns.color_palette("hls", 2),
    data=data.loc[:,:],
    legend="full",
    alpha=0.3
)
plt.show()




