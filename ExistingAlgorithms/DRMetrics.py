import pandas as pd
import numpy as np
import math
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class matrix_metrics:
    """
    [1] Y. Zhang, Q. Shang, and G. Zhang, 
    ‘pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment’, 
    Heliyon, vol. 7, no. 2, p. e06199, Feb. 2021, doi: 10.1016/j.heliyon.2021.e06199.
    """
    def __init__(self, X_list, Z_list, Xr_list, Titles):

        self.min_cluster = 30
        self.max_cluster = 50

        metric_list = [
            'Reconstruction Error'            ,
            'Relative Reconstruction Error'   ,
            'Residual variance Pearson'       , 
            'Residual variance Spearman'      ,
            'Trustworthiness'                 ,
            'Continuity'                      ,
            'Co-k-nearest neighbor size'      , 
            'The area under the curve'        , 
            'Local Continuity Meta Criterion' , 
            'Local property metric'           ,
            'Global property metric'          ,

            'silhouette kmeans'               , 
            'calinski harabasz kmeans'        ,
            'davies bouldin kmeans' 
        ]


        scores = []
        for index, name in tqdm(enumerate(Titles), total=len(Titles)):
            scores.append(self.metrics(X_list[index], Z_list[index], Xr_list[index]))

        plt.figure(figsize=(10,5))
        plt.imshow(scores,aspect='auto')#, interpolation="bilinear")
        plt.xlabel('Metric')
        plt.ylabel('Method')
        plt.xticks(np.arange(len(metric_list)), labels=[i[0] for i in metric_list])
        plt.yticks(np.arange(len(Titles)), labels=Titles)
        #plt.colorbar()

        for (j,i),label in np.ndenumerate(scores):
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')
            plt.text(i,j,'{:.2e}'.format(label),ha='center',va='center')

        plt.show()

            




        

    def metrics(self, X, Z, Xr):
            '''
            Initialization. X is the original data. Z is the data after DR. Xr is the reconstructed data.
            X : Data before DR. m-by-n matrix
            Z : Data after DR. m-by-k matrix. Typically, k << n
            '''
            
            # print(X.shape, Z.shape)
            assert X.shape[0] == Z.shape[0]
                    
            if Xr.any() == None:
                mse, ms, rmse = None, None, None
            else:
                assert X.shape == Xr.shape
                mse, ms, rmse = self.calculate_recon_error(X, Xr)
                
                
                
            # Construct the distance matrix
            print("Construct the distance matrix")
            df = pd.DataFrame(X, index=None)
            D = pd.DataFrame(distance_matrix(df.values, df.values)).values        

            dfz = pd.DataFrame(Z, index=None)
            Dz = pd.DataFrame(distance_matrix(dfz.values, dfz.values)).values
            
            # Residual Variance of the two distance matrices 
            print("Residual Variance of the two distance matrices")
            Vr = 1 - (pearsonr(D.flatten(), Dz.flatten())[0])**2 # Pearson's r version
            Vrs = 1 - (spearmanr(D.flatten(), Dz.flatten())[0])**2 # Spearman' r version

            R = self.ranking_matrix(D)
            Rz = self.ranking_matrix(Dz)
            Q = self.coranking_matrix(R, Rz)
        
            print("Coranking matrix metrics")
            T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal = self.coranking_matrix_metrics(Q)

            AUC_T = np.mean(T)
            AUC_C = np.mean(C)

            SK = self.silhouette_kmeans(Z)
            SCH = self.calinski_harabasz_kmeans(Z)
            SDB = self.davies_bouldin_kmeans(Z)


            results = np.array([
                mse ,
                rmse ,
                Vr , 
                Vrs ,
                T ,
                C ,
                QNN , 
                AUC , 
                LCMC , 
                Qlocal ,
                Qglobal,

                SK , 
                SCH ,
                SDB 
            ])

            return results



    def ranking_matrix(self, D):    
        D = np.array(D)    
        R = np.zeros(D.shape)
        m = len(R)

        for i in range(m):
            for j in range(m):
                Rij = 0
                for k in range(m):
                    if (D[i,k] < D[i,j]) or (math.isclose(D[i,k], D[i,j]) and k < j ):
                        Rij += 1
                R[i,j] = Rij
                
        return R

    '''
    R1, R2 - two ranking matrices
    '''
    def coranking_matrix(self, R1, R2):
        
        R1 = np.array(R1)
        R2 = np.array(R2)    
        assert R1.shape == R2.shape    
        Q = np.zeros(R1.shape)
        m = len(Q)

        #for k in tqdm(range(m)): # Constructing Q is the most time-consuming process
        #    for l in range(m):
        #        kl = 0
        #        for i in range(m):
        #            for j in range(m):
        #                if R1[i,j] == k and R2[i,j] == l:
        #                    kl += 1
        #        Q[k,l] = kl

        for i in tqdm(range(m), total=m):
            for j in range(m):
                k = int(R1[i,j])
                l = int(R2[i,j])
                Q[k,l] += 1

        return Q


    def coranking_matrix_metrics(self, Q):
        
        Q = Q[1:,1:]
        m = len(Q)
        
        T = np.zeros(m-1) # trustworthiness
        C = np.zeros(m-1) # continuity
        QNN = np.zeros(m) # Co-k-nearest neighbor size
        LCMC = np.zeros(m) # Local Continuity Meta Criterion
        
        for k in tqdm(range(m-1) , total=m):
            Qs = Q[k:,:k]
            W = np.arange(Qs.shape[0]).reshape(-1, 1) # a column vector of weights. weight = rank error = actual_rank - k
            T[k] = 1-np.sum(Qs * W)/(k+1)/m/(m-1-k)  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
            Qs = Q[:k,k:]
            W = np.arange(Qs.shape[1]).reshape(1, -1) # a row vector of weights. weight = rank error = actual_rank - k
            C[k] = 1-np.sum(Qs * W)/(k+1)/m/(m-1-k)  # 1 - normalized hard-k-extrusions. upper-right region 
        
        for k in tqdm(range(m), total=m):    
            QNN[k] = np.sum(Q[:k+1,:k+1])/((k+1) * m) # Q[0,0] is always m. 0-th nearest neighbor is always the point itself. Exclude Q[0,0]
            LCMC[k] = QNN[k] - (k+1)/(m-1)

        kmax = np.argmax(LCMC)
        Qlocal = np.sum(QNN[:kmax+1])/(kmax + 1)
        Qglobal = np.sum(QNN[kmax:-1])/(m - kmax -1) # skip the last. The last is (m-1)-nearest neighbor, including all samples.
        AUC = np.mean(QNN)
        
        return T, C, QNN, AUC, LCMC, kmax, Qlocal, Qglobal
    
    def recon_MSE(self, X,Xr):
        assert X.shape == Xr.shape
        return norm(X-Xr, ord='fro')**2/(X.shape[0]*X.shape[1])


    def calculate_recon_error(self, X, Xr):
        assert X.shape == Xr.shape

        mse = self.recon_MSE(X, Xr) # mean square error
        ms = self.recon_MSE(X, np.zeros(X.shape)) # mean square of original data matrix

        return mse, ms, mse/ms
    



    def score_kmeans(self, Z, score):
        eval_cluster = range(self.min_cluster+1 , self.max_cluster+1)
        temp_score = []

        for cluster_index, cluster_number in enumerate(eval_cluster):
                
                clusters = KMeans(n_clusters=cluster_number, random_state=42, init="k-means++", n_init='auto').fit_predict(Z[::10])
                temp_score.append(score(Z[::10], clusters))

        opt_n_clusters = eval_cluster[np.argmax(temp_score)]
        opt_clusters = KMeans(n_clusters=opt_n_clusters, random_state=42, init="k-means++", n_init='auto').fit_predict(Z[::10])

        return score(Z[::10], opt_clusters)
    

    def silhouette_kmeans(self, Z):
         return self.score_kmeans(Z, silhouette_score)

    def calinski_harabasz_kmeans(self, Z):
        return self.score_kmeans(Z, calinski_harabasz_score)
    
    def davies_bouldin_kmeans(self, Z):
        return self.score_kmeans(Z, davies_bouldin_score)