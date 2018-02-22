from sklearn.decomposition import PCA
import dataset

dir = './'

data = dataset.read_data_sets(dir, one_hot=True, validation_size=0, test_size=0, payload_length=810).train
datapoints = data.payloads

pca = PCA(n_components=0.9, svd_solver='full')
pca.fit(datapoints)
print(pca.n_components_)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))