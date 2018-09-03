# Oguz Gul - 220201015
import sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from time import time

def read_data():
    df = pd.read_csv('53727641925dat.txt', delim_whitespace=True, low_memory=False)
    cols = list(df.columns.values)
    print(cols)
    print(len(cols))
    enc  = LabelEncoder()
    for col in cols:
        df[col] = enc.fit_transform(df[col].values.astype('U'))
    return df
def pca_dr(df,y):
    df = StandardScaler().fit_transform(df)
    pcal = PCA(n_components=0.99, svd_solver='full')
    t0 = time()
    reduced = pcal.fit_transform(df,y)
    t1 = time()
    plt.scatter(reduced[:, 0], reduced[:, 1],edgecolor='none',c=reduced[:, 0], cmap=plt.cm.Spectral, alpha=0.5)
    plt.title("PCA: %.2g sec" % (t1-t0))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
    print("With given n_components=0.99 ", pcal.n_components_," components(dimensions) calculated. Eventhough cumulative variance could not pass:, ", pcal.explained_variance_.cumsum())

def mds(df,y):
    df = df.iloc[:1000]
    mds = MDS(n_components=2, dissimilarity='euclidean',n_jobs=2, n_init=2 ,max_iter=100)
    t0 = time()
    reduced = mds.fit_transform(df,y)
    t1 = time()
    plt.scatter(reduced[:,0],reduced[:,1], edgecolor='none',c=reduced[:,0],cmap=plt.cm.Spectral)
    plt.title("MDS: %.2g sec" % (t1-t0))
    plt.xlabel('X axis')
    plt.ylabel('Y Axis')
    plt.show()
    print("Final value of stress in MDS: ",mds.stress_)

def isomap(df,y):
    df  = df.iloc[:2000]
    iso = Isomap(n_components=3,path_method='D',neighbors_algorithm='auto',eigen_solver='auto',n_jobs=2,n_neighbors=7)
    t0 = time()
    reduced = iso.fit_transform(df,y)
    t1 = time()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=reduced[:, 0], cmap=plt.cm.Spectral)
    plt.title("Isomap: %.2g sec" % (t1 - t0))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
    print("Geodesic distance matrix in ISOMAP Algorithm: ", iso.dist_matrix_)

def lle(df,y):
    df  = df.iloc[:2000]
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5, eigen_solver='auto', method='standard', neighbors_algorithm='auto',n_jobs=2)
    t0  = time()
    reduced = lle.fit_transform(df)
    t1  = time()
    fig = plt.figure()
    plt.scatter(reduced[:,0],reduced[:,1],edgecolor='none', c=reduced[:,0],cmap=plt.cm.Spectral)
    plt.title("Locally Linear Embedding: %.2g sec" % (t1-t0))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()
    print("Reconstruction error associated with embedding vectors is: ", lle.reconstruction_error_)
def main():
    df = read_data()
    y = df.TEMP
    X  = df.drop('TEMP', axis=1)
    pca_dr(X,y)
    isomap(X,y)
    mds(X,y)
    lle(df,y)
if __name__ == "__main__":
    main()
