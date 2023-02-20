import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000, 100).T
for col in range(data.shape[0]):
    if np.std(data[col]) == 0: continue
    else: data[col] = (data[col] - np.mean(data[col])) / np.std(data[col])
standaradized_data = data.T

U, S, VT = np.linalg.svd(standaradized_data, full_matrices=False)
# print(U.shape, S.shape)
princ_comp = np.dot(U, S)

# for i in range(2):
currVector = princ_comp[:2] # eigenvectors
proj_matrix = (np.dot(currVector, currVector.T) / np.dot(currVector.T, currVector)) * standaradized_data
print(proj_matrix.shape)


for y in proj_matrix:
    plt.scatter(np.arange(0,100), y)

plt.show()


'''

from sklearn.decomposition import PCA

pca = PCA()
#
# Determine transformed features
#
X_train_pca = pca.fit_transform(standaradized_data)
#
# Determine explained variance using explained_variance_ration_ attribute
#
exp_var_pca = pca.explained_variance_ratio_
#
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
#
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
#
# Create the visualization plot
#
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

'''
