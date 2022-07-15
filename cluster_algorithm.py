import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'  # Constrain the number of core to execute numpy
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
import numpy as np                # Calculate the inner product
from sklearn.cluster import AffinityPropagation, KMeans, kmeans_plusplus, DBSCAN, OPTICS

def cluster_by_affi(byte_array_series_run):
    S = -np.square(pairwise_distances(byte_array_series_run)) #Affinity matrix
    prefer = np.mean(S) #set preference value
    model = AffinityPropagation(preference = prefer)
    model.fit(byte_array_series_run)
    labels = model.labels_
    cluster_center = model.cluster_centers_

    return model, labels, cluster_center

# K-means Propagation method
def cluster_by_kmeans(byte_array_series_run, n_clusters, n_run):

    model = KMeans(n_clusters=n_clusters, n_init = n_run).fit(byte_array_series_run)
    labels = model.labels_
    cluster_center = model.cluster_centers_
    inertia = model.inertia_
    print(f'Inertia: {inertia}')

    return model, labels, cluster_center, inertia



def cluster_by_dbscan(byte_array_series_run, radius, min_samples):
  
    model = DBSCAN(eps=radius, min_samples=min_samples).fit(byte_array_series_run)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    cluster_center = model.components_

    return labels, cluster_center, n_clusters, n_noise

def cluster_by_optics(byte_array, min_samples, stepness, max_eps):
    model = OPTICS(min_samples=min_samples, xi=stepness, max_eps=max_eps).fit(byte_array)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return labels, n_clusters, n_noise


# K-means Propagation method
def cluster_by_kmeans_random(byte_array_series_run, n_clusters):
    model = KMeans(n_clusters=n_clusters, max_iter=500, n_init=1).fit(byte_array_series_run)
    labels = model.labels_
    cluster_center = model.cluster_centers_
    inertia = model.inertia_
    print('Inertia: ' + str(inertia))
    print('Iteration: ' + str(model.n_iter_))

    return model, labels, cluster_center
  
  if __name__ == '__main__':
    data_list = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 4, 6]])
    labels, n_clusters, n_noise = cluster_by_optics(data_list, 2, 0.01, 5)
