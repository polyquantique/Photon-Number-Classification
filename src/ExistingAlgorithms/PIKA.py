import numpy as np
import math
from tqdm.notebook import tqdm
from scipy.stats import poisson
from scipy.special import gamma, loggamma
from numpy.random import shuffle, choice
import matplotlib.pyplot as plt
from os import listdir


from scipy.signal import savgol_filter, wiener


class PIKA():
    
    def __init__(self, config):
        """
        TODO : Finish implementation
        """
        # Dataset 
        self.folder = config['Dataset_path']
        self.size = config['Dataset_signal_size']
        
        # Initial assumptions
        #self.N_mean_list = config['Average_photon_number']
        
        # Loop iteration 
        self.optimization_iter = config['Optimization_iter']
        self.nb_cluster = 0
        
        # Normalization
        self.objective_sigma = 0

        # Signal parameters
        self.nb_samples = 0
        self.N_mean = config['Average_photon_number']
        self.size_clusters = []
        self.V_mean = []

        # Objective history
        self.O_K = []
        self.O_PC = 0

        # Gradient descent implementation
        self.epochs = config['Epochs']


        
        

    def Get_traces(self):

        V = np.concatenate([np.fromfile(f"{self.folder}/{file_name}", dtype=np.float16).reshape((-1,self.size)) for file_name in listdir(self.folder)])
        self.nb_samples = len(V)
        index = np.arange(self.nb_samples)

        V = savgol_filter(V, 10, 2)
        #V = wiener(V, (5, 5))


        return V.astype(float), index





    def Dot_product(self, V, index):
        V_mean = np.mean(V, axis = 0)
        n_eff = self.N_mean * np.sum(V_mean * V, axis=1) / np.sum(V_mean * V_mean)

        _, index = zip(*sorted(zip(n_eff, index)))

        return index
    




    def Poisson(self, index):
        # Compute 95% confidence interval
        sigma = np.sqrt(self.N_mean)
        upper_limit = math.ceil(self.N_mean + 1.3 * sigma)
        lower_limit = math.floor(self.N_mean - 1.3 * sigma)
        if lower_limit < 0: lower_limit = 0

        N = np.arange(lower_limit, upper_limit + 1)
        prob = poisson.pmf(N, self.N_mean)
        #prob = np.absolute( np.exp(-self.N_mean) * self.N_mean**N / gamma(N + 1) )
        int_prob = np.rint(prob / np.sum(prob) * self.nb_samples).astype('int64')

        N = N[int_prob > 0]
        int_prob = int_prob[int_prob > 0]
        

        int_prob[np.argmax(int_prob)] += self.nb_samples - np.sum(int_prob)

        split = [int(np.sum(int_prob[:i+1])) for i, _ in enumerate(int_prob)]

        return N , np.split(index, split)[:-1]





    def Initial_cluster(self, V, index):
        index = self.Dot_product(V, index)
        N, clusters = self.Poisson(index)
        self.nb_cluster = len(clusters)
        
        return N, clusters





    def Initialization(self, V, clusters):

        self.size_clusters = [] # Initialization of size_clusters
        self.V_mean = []
        O_KPC_list = []

        for i in clusters:
            self.size_clusters.append(len(i))
            i = i.reshape(-1,1)
            self.V_mean.append(np.mean(V[i][0], axis=0))

        self.V_mean = np.array(self.V_mean)
        self.size_clusters = np.array(self.size_clusters)
        
        return O_KPC_list
    




    def KMeans_objective(self, V, clusters):   
        O_K = []
        for index_cluster, cluster in enumerate(clusters):
            O_K.append((1/self.size) * np.sum((V[cluster] - self.V_mean[index_cluster])**2))
        #x = np.std(V)
        self.objective_sigma = np.ones(self.nb_cluster) * np.sqrt(sum(O_K) / self.nb_cluster)  # Definition of objective_sigma

        return O_K





    def Log_Poisson_likelihood(self, N, len_clusters):
        L_P = - self.N_mean * np.sum(len_clusters) \
              + np.sum(len_clusters * (N * np.log(self.N_mean) - loggamma(len_clusters + 1)))
        return L_P

    def Log_Combinatorial_likelihood(self, len_clusters):
        L_C = loggamma(np.sum(len_clusters)+1) - np.sum(loggamma(len_clusters+1))
        return L_C

    def Poisson_Combinatorial_objective(self, N):
        O_PC = self.Log_Poisson_likelihood(N, self.size_clusters) + self.Log_Combinatorial_likelihood(self.size_clusters)
        return - O_PC # Log of the likelihoods





    
    def KMeans_Poisson_objective(self, O_K, O_PC):
        O_KPC = 1/(2*self.objective_sigma[0]**2) * sum(O_K) + O_PC
        return O_KPC





    def Randomize_clusters(self, clusters):
        for index, cluster in enumerate(clusters):
            shuffle(cluster)
            clusters[index] = cluster

        return clusters





    def Get_neighbors(self, N):

        len_ = len(N)
        neighbors = []
        for index in range(len_):
            neighbors.append([index-1, index+1])
        neighbors[0] = [1]
        neighbors[-1] = [len_-2]
        
        return neighbors
 




    def meanSquare(self, waveform, cluster_idx):
            return 1/self.size * np.sum((waveform - self.V_mean[cluster_idx])**2)

    def Update_KMean(self, waveform, add_idx, del_idx):

        size_add = self.size_clusters[add_idx]
        size_del = self.size_clusters[del_idx]

        delta_add = size_add / (size_add + 1) * self.meanSquare(waveform, add_idx)
        delta_del = size_del / (size_del - 1) * self.meanSquare(waveform, del_idx)

        delta_kmeans_scaled = size_add / (2 * self.objective_sigma[add_idx]**2) \
                            - size_del / (2 * self.objective_sigma[del_idx]**2)

        return delta_add, delta_del, delta_kmeans_scaled





    def Delta_Poisson(self, N, add_idx, del_idx):
        return (N[add_idx]-N[del_idx]) * np.log(self.N_mean) - loggamma(N[add_idx]+1) + loggamma(N[del_idx]+1)

    def Delta_Combinatorial(self, add_idx, del_idx):
        return np.log(self.size_clusters[del_idx] / (self.size_clusters[add_idx] + 1))

    def Update_Poisson_Combinatorial(self, N, add_idx, del_idx):
        return -(self.Delta_Poisson(N, add_idx, del_idx) + self.Delta_Combinatorial(add_idx, del_idx))
        




    def Update_Param(self, clusters, waveform, inner_cluster_idx, add_idx, del_idx, del_wave_idx):

        self.V_mean[add_idx] = (self.size_clusters[add_idx] * self.V_mean[add_idx] + waveform)/(self.size_clusters[add_idx] + 1)
        self.V_mean[del_idx] = (self.size_clusters[del_idx] * self.V_mean[del_idx] - waveform)/(self.size_clusters[del_idx] - 1)

        self.size_clusters[add_idx] += 1
        self.size_clusters[del_idx] -= 1

        clusters[add_idx] = np.append(clusters[add_idx] , clusters[del_idx][inner_cluster_idx])

        clusters[del_idx] = np.delete(clusters[del_idx] , inner_cluster_idx)
        del_wave_idx += 1

        return clusters, del_wave_idx


    


    

    def run_PIKA(self, plot=False):

        N_mean_list = []
        O_KPC_list_epoch = []

        V, index = self.Get_traces()
        
        for epoch in tqdm(range(self.epochs), desc="Epoch"):

            N, clusters = self.Initial_cluster(V, index)    # Dot product alg. 

            O_KPC_list = self.Initialization(V, clusters)

            O_K = self.KMeans_objective(V, clusters)    # Initialization of K-means objective function
            O_PC = self.Poisson_Combinatorial_objective(N)    # Initialization of Poisson-Combinatorial objective function

            O_KPC_list.append(self.KMeans_Poisson_objective(O_K, O_PC))   # Initialization of objective function
            neighbors = self.Get_neighbors(N)

            for opt_idx in range(self.optimization_iter):   # Number of shuffles (User input)

                clusters = self.Randomize_clusters(clusters)    # Randomize elements inside clusters

                for cluster_idx , cluster in enumerate(clusters):  # Iterate through all clusters 
                    del_wave_idx = 0
                    for inner_cluster_idx, waveform_idx in enumerate(cluster):
                        inner_cluster_idx = inner_cluster_idx - del_wave_idx
                        waveform = V[waveform_idx]

                        if self.size_clusters[cluster_idx] > 2:
                            add_idx = choice(neighbors[cluster_idx])
                            del_idx = cluster_idx

                            temp_O_K_add, temp_O_K_del, delta_kmeans_scaled = self.Update_KMean(waveform, add_idx, del_idx)
                            temp_O_PC = self.Update_Poisson_Combinatorial(N, add_idx, del_idx)

                            if delta_kmeans_scaled + temp_O_PC < 0:
                                O_K[add_idx] += temp_O_K_add
                                O_K[del_idx] -= temp_O_K_del
                                O_PC += temp_O_PC
                    
                                O_KPC_list.append(O_KPC_list[-1] + delta_kmeans_scaled + temp_O_PC)

                                clusters, del_wave_idx = self.Update_Param(clusters, waveform, inner_cluster_idx, add_idx, del_idx, del_wave_idx)

    
            alpha = 1e-5
            if len(O_KPC_list_epoch) == 0:
                O_KPC_list_epoch.append(O_KPC_list[0])
                N_mean_list.append(self.N_mean)

            O_KPC_list_epoch.append(O_KPC_list[-1])
            N_mean_list.append(self.N_mean)

            self.N_mean = N_mean_list[-1] - alpha * (O_KPC_list_epoch[-1] - O_KPC_list_epoch[-2]) / N_mean_list[-2]

        if plot:
            
            # Last O_KPC
            plt.figure()
            plt.plot(O_KPC_list, label = r"$O_{KPC}$")
            plt.xlabel("Successfull move")
            plt.ticklabel_format(useOffset=False)
            plt.legend()
            plt.show()
            
            # Mean traces for each photon number
            plt.figure()
            pl = [plt.plot(wave, label=f"{N[index]}") for index, wave in enumerate(self.V_mean)]
            plt.ticklabel_format(useOffset=False)
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(N_mean_list)
            plt.ticklabel_format(useOffset=False)
            plt.ylabel("Average photon number")

            plt.figure()
            plt.ticklabel_format(useOffset=False)
            plt.plot(O_KPC_list_epoch)
            plt.ylabel(r"$O_{KPC}$")

        return clusters