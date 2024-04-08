import pandas as pd
from sklearn.cluster KMeans
from matplotlib import pyplot as plt
from EvaluationMatrices import EvaluationMatrices
from sklearn.metrics import silhouette_score
import numpy as np
from minisom import MiniSom
import warnings
warnings.filterwarnings("ignore")



def doSOM(X_train, max_k):

    X_train_df = X_train.copy()
    X_train_df = X_train_df[X_train_df.columns[14:-1]]
    TC = []
    sil = []

    for k in range(2, max_k):

        #linear som topography
        som_shape = (1, k)

        som = MiniSom(som_shape[0], som_shape[1], X_train_df.shape[1], sigma=0.5, learning_rate=0.5)

        max_iter = 1000
        q_error = []

        for i in range(max_iter):
            rand_i = np.random.randint(len(X_train_df))
            som.update(X_train_df[rand_i], som.winner(X_train_df[rand_i]), i, max_iter)
            q_error.append(som.quantization_error(X_train_df))

        # each neuron represents a cluster
        winner_coordinates = np.array([som.winner(x) for x in X_train_df]).T
        # convert the bidimensional
        # coordinates to a monodimensional index
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
        sil.append(silhouette_score(X_train_df, cluster_index))
        X_train['clusters'] = cluster_index
        TC.append(evaluationMatrices.calcTreeTaxonomy(X_train[list(X_train.columns[2:11]) + ['clusters']]))

        # plt.plot(np.arange(max_iter), q_error, label='quantization error')
        # plt.plot(np.arange(max_iter), t_error, label='topographic error')
        # plt.ylabel('Quantization error')
        # plt.xlabel('Iteration index')
        # plt.legend()
        # plt.show()


def ClusterEachTaxonomy(df, rank):
    '''
    Visualization of each suggested cluster within each NCBI taxonomy
    '''

    for cluster in df['clusters'].unique():
        x = []
        y = []
        d = df.copy()
        d = d[d['clusters'] == cluster]

        for tax in d[rank].unique():
            x.append(tax)
            y.append(len(d[d[rank] == tax]))

        plt.bar(x,y)
        plt.title(f"histogram of taxonomy in cluster {cluster} for {rank}")
        plt.xlabel("taxonomy")
        plt.ylabel("count")
        plt.savefig(f"CET {cluster} {rank}.jpg")
        plt.show()

def TaxonomyEachCluster(df , rank):

    '''
    Visualization of NCBI's taxonomy within each suggested cluster
    '''

    for tax in df[rank].unique():
        x = []
        y = []
        d = df.copy()
        d = d[d[rank] == tax]

        for cluster in d['clusters'].unique():
            x.append(cluster)
            y.append(len(d[d['clusters'] == cluster]))

        plt.bar(x, y)
        plt.title(f"histogram of clusters in taxonomy {tax} for {rank}")
        plt.xlabel("cluster")
        plt.ylabel("count")
        plt.savefig(f"TEC {tax} {rank}.jpg")
        plt.show()


def doKMeans(X_train, max_k):
    '''
    This function will find the optimal split (based on TC score)
    '''

    X_train_df = X_train.copy()
    X_train_df = X_train_df[X_train_df.columns[14:-1]]
    TC = []
    sil = []

    for i in range(2,max_k):

        print(f'fitting clusters = {i}')
        kmeans = KMeans(n_clusters=i).fit(X_train_df)
        sil.append(silhouette_score(X_train,kmeans.labels_))
        X_train['clusters'] = kmeans.labels_
        TC.append(evaluationMatrices.calcTreeTaxonomy(X_train[list(X_train.columns[2:11]) + ['clusters']]))
        print(f'score is : {TC[-1]}')

    print(f'min arg is {np.argmin(TC)}')
    print(TC)
    kmeans = KMeans(n_clusters=(np.argmin(TC) + 2)).fit(X_train_df)
    X_train_df['clusters'] = kmeans.labels_
    ClusterEachTaxonomy(X_train_df, 'rank_9')
    TaxonomyEachCluster(X_train_df, 'rank_9')
    X_train_df.to_csv("first optimal split.csv")


evaluationMatrices = EvaluationMatrices()
data = pd.read_csv("ready_to_run_codon.csv")
doKMeans(data,9)
doSOM(data,9)
