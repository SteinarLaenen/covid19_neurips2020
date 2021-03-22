import csv
import pandas as pd
import numpy as np
from spectralDirectedClustering import DirectedSpectralClustering
from spectralClustering import SpectralClustering
import scipy.linalg as linalg
import copy
import json
from matplotlib.patches import Patch
from itertools import permutations

import matplotlib.pyplot as plt

rcParams = {
    'axes.labelsize': 35,
#    'suptitle.fontsize': ,
    'font.size': 32,
    'lines.linewidth': 5,
    'lines.marker': '.',
    'lines.markeredgewidth': 20,
    'legend.fontsize': 60,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'figure.figsize': [30, 15],
    'grid.linewidth': 0.0,
    'figure.subplot.hspace': 0.08,
    'figure.subplot.wspace': 0.08
}

plt.rcParams.update(rcParams)

plot_colors = ['r', 'g', 'b', 'm', 'c', 'k', 'y']

def main():
    data_file = './PatientInfo.csv'
    
    # obtain patient information
    patient_list, patient_ids, idx_to_patient_id_dic, patient_id_to_idx_dic = get_patient_info(data_file)    

    # create adjacency matrix
    M, remove_zeros_dic = create_adjacency_matrix(patient_list, patient_id_to_idx_dic, idx_to_patient_id_dic)
    print("num edges", np.sum(np.sum(M, axis=0), axis=0))

    idx = get_largest_connected_component(M)

    M = M[:, idx]
    M = M[idx, :]
    print("num nodes", M.shape[0])
    print("num edges", np.sum(np.sum(M, axis=0), axis=0))
    
    
    n = M.shape[0]

    # Add some noise to the adjacency matrix for regularisation (uncomment the noise)
    M_noise = M# + np.triu(np.random.rand(len(M), len(M))*0.01)

    # define k and omega k
    k = 4
    omega_k = np.cos((2*np.pi)/np.ceil(2*np.pi*k)) + 1j*np.sin((2*np.pi)/np.ceil(2*np.pi*k))
    print("Abs", np.abs(omega_k))

    # make the hermitian adjacency matrix
    Herm = M_noise*omega_k + M_noise.T*np.conj(omega_k)

    # compute the diagonal matrix
    D = np.identity(n)*np.sum(np.abs(M_noise + M_noise.T), axis=0)
    print("total edges in graph = ", np.sum(D)/2)

    # compute the inverse of the diagonal matri
    d_norm = linalg.fractional_matrix_power(D, -0.5)
    
    #norm_Laplacian = Laplacian
    # Create the normalized Laplacian
    Laplacian = D - Herm
    norm_Laplacian = np.matmul(np.matmul(d_norm, Laplacian), d_norm)

    print("IS HERMITIAN:", np.allclose(norm_Laplacian, np.conj(norm_Laplacian.T), rtol=1e-8, atol=1e-8))
    
    eigvals_norm, eigvects_norm = np.linalg.eigh(norm_Laplacian)
    print("SMALLEST EIGVALS:", np.sort(eigvals_norm)[0:5])
    
    print("number disconnected components (estimate):", np.sum(np.real(np.sort(eigvals_norm))<0))

    d_norm = np.sum(d_norm, axis=0)

    spectral_labels_herm, eigvals, eigvects, W = spectral_clustering(norm_Laplacian, d_norm, k, "herm", n_eigvects=1, preload=False)

    print(min(spectral_labels_herm))
    print(max(spectral_labels_herm))
    print("len spectral labels", len(spectral_labels_herm), len(eigvals), eigvects.shape)

    # np.save("spectral_labels_herm.npy", spectral_labels_herm)
    # np.save("eigvals.npy", eigvals)
    # np.save("eigvects.npy", eigvects)

    # spectral_labels_herm = np.load("spectral_labels_herm.npy")
    # eigvals = np.load("eigvals.npy")
    # eigvects = np.load("eigvects.npy")
    print("Spectral Gap between lamda1 and lamda 2:", np.real(np.sort(eigvals_norm))[1]/np.real(np.sort(eigvals_norm))[0])

    clusters = [[j for j in range(len(spectral_labels_herm)) if spectral_labels_herm[j] == i] for i in range(k)]

    indices = eigvals.argsort()
    lamda1 = eigvals[indices[0]]
    print("First eigenvalues:", lamda1)
    top1_eigvec = eigvects[:,indices[0]]


    new_ordering = compute_master_flow(M, spectral_labels_herm, k)


    print("number of patients in each cluster:")
    for i in range(k):
        cluster = clusters[i]
        print "length of cluster " + str(i) + "=" + str(len(cluster))
    
    
    for cl1 in range(k):
        for cl2 in range(k):
            net_flow = compute_flow_between_clusters(M, clusters[cl1], clusters[cl2])
            print "net flow between: " + str(new_ordering.index(cl1)) + " and " + str(new_ordering.index(cl2)) + " = " + str(net_flow)

    x = W[:,0]#np.real(top1_eigvec)
    y = W[:,1]#np.imag(top1_eigvec)
    for cl_idx in range(k):
        plt.scatter(x[clusters[cl_idx]],y[clusters[cl_idx]], s=700,
                    label="S" + str(new_ordering.index(cl_idx)),
                    color=plot_colors[new_ordering.index(cl_idx)])

    legend_elements = []
    for cl_idx in range(k):
        patch = Patch(facecolor=plot_colors[cl_idx], edgecolor='w', label='S' + str(cl_idx))
        legend_elements.append(patch)
        
    # legend_elements = [Patch(facecolor='red', edgecolor='w',
    #                          label='S 1'),
    #                    Patch(facecolor='0.3', edgecolor='w',
    #                          label='S 2'),
    #                    Patch(facecolor='0.7', edgecolor='w',
    #                          label='S 3'),
    #                    Patch(facecolor='#f8c034', edgecolor='w',
    #                          label='cluster 4')] 
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.show()
#    plt.savefig('./neurips_plots/covid19.pdf')


    # compute how many people were infected by each person
    how_many_infected = np.zeros(n)
    for patient in patient_list:
        infected_by =  patient.infected_by
        try:
            patient_idx = patient_id_to_idx_dic[infected_by]
            how_many_infected[patient_idx] += 1
        except:
            continue

    print("super spreaders:", D.sum(0))

    # for patient_idx in clusters[0]:
    #     patient = patient_list[patient_idx]
    #     print(patient.infected_by)
#        print(len(clusters[0]))


def get_largest_connected_component(M):
    adjacency_matrix = M + M.T
    D = np.sum(adjacency_matrix, axis=0)
    n = adjacency_matrix.shape[0]

    max_visited = []
    visited = []
    label = 0
    for idx in range(n):
        if not idx in visited:
            visited.append(idx) 
            dfs(idx, adjacency_matrix, visited, label)
            if len(visited) > len(max_visited):
                print("prev max:", len(visited))
                max_visited = copy.deepcopy(visited)

            visited = []

    return max_visited

def compute_master_flow(adj_matrix, spectral_labels, k):
    n_clusters = max(spectral_labels) + 1
    cluster_pair_flow_dic = {}
    for cluster1 in range(n_clusters):
        for cluster2 in range(n_clusters):
            if cluster1 != cluster2:
                cluster1_idx = [j for j in range(len(spectral_labels)) if spectral_labels[j]==cluster1]
                cluster2_idx = [j for j in range(len(spectral_labels)) if spectral_labels[j]==cluster2]

                W_cl1_to_cl2 = 0
            
                for i in cluster1_idx:
                    for j in cluster2_idx:
                        W_cl1_to_cl2 += adj_matrix[i,j]

                        
                cluster_pair_flow_dic[(cluster1, cluster2)] = W_cl1_to_cl2

    all_paths = list(permutations(range(n_clusters)))
    max_masterflow = 0
    for path in all_paths:
        master_flow = 0
        for i in range(n_clusters-1):
            master_flow += cluster_pair_flow_dic[(path[i], path[i+1])]

        if master_flow > max_masterflow:
            max_masterflow = master_flow
            print(max_masterflow)
            max_masterpath = path

            
    return max_masterpath

    
 
def dfs(idx, adjacency_matrix, visited, label):
    neighbours = adjacency_matrix[idx]
    idx_neighbours = np.where(adjacency_matrix[:, idx] == 1)[0]
    for idx in idx_neighbours:
        if not idx in visited:
            visited.append(idx)
            dfs(idx, adjacency_matrix, visited, label)
    return
    
    

def compute_flow_between_clusters(adjacency_matrix, cluster_labels_1, cluster_labels_2):
    W_cl1_to_cl2 = 0
    W_cl2_to_cl1 = 0
    for i in cluster_labels_1:
        for j in cluster_labels_2:
            W_cl1_to_cl2 += adjacency_matrix[i,j]
            W_cl2_to_cl1 += adjacency_matrix[j,i]

    return W_cl1_to_cl2 - W_cl2_to_cl1


def spectral_clustering(A, d_norm, n_clusters, case, n_eigvects=1, lamda=0, preload=False):
    """ performs spectral clustering and returns some values
    """
    # if A is hermitian, perform the complex valued case
    if case=="herm":
        spectral_labels, eigvals, \
        eigvects, W = DirectedSpectralClustering(n_clusters,
                                                 A,
                                                 "UnnormalizedLaplacianMatrix"
                                                 ,n_eigvects, d_norm, preload=preload)




    else:
        spectral_labels, eigvals_normal, \
        eigvects_normal, W_normal = SpectralClustering(n_clusters,
                                                       A, "AdjacencyMatrix")

    return spectral_labels, eigvals, eigvects, W

def create_adjacency_matrix(patient_list, patient_id_to_idx_dic, idx_to_patient_id_dic, remove_zeros=True):
    n = len(patient_list)
    M = np.zeros((n,n))

    for patient in patient_list:
        patient_idx = patient.idx_id
        infected_by_patient_id = patient.infected_by
        if len(infected_by_patient_id) > 0:
            # if infection by whom is known, extract their id and enter entry in adj matrix
            # try except statement because sometimes patient info about global number is not know
            try:
                infected_by_patient_idx = patient_id_to_idx_dic[infected_by_patient_id]
            except:
                continue

            M[infected_by_patient_idx][patient_idx] = 1

            
    remove_zeros_dic = {}
    counter = 0
    X = M + M.T
    if remove_zeros:
        idx_to_keep = []
        for i in range(n):
            if ~np.all(X[i, :] == 0):# or ~np.all(M[:, i] == 0):
                idx_to_keep.append(i)
                remove_zeros_dic[counter] = i
                counter+=1
                

        M = M[idx_to_keep]
        M = M[:, idx_to_keep]

    return M, remove_zeros_dic

def get_patient_info(data_file):
    patient_list = []
    with open(data_file, mode='rb') as patientinfo:
        reader = csv.DictReader(patientinfo)

        patient_ids = []
        for i, row in enumerate(reader):
            patient = Patient(row['patient_id'],
                              i,
                              row['sex'],
                              row['infected_by'],
                              row['birth_year'],
                              row['confirmed_date'],
                              row['city'])

            patient_ids.append(row['patient_id'])
            patient_list.append(patient)



    # We make the patient list sparser, to remove patients with no in or outgoing edges
    print(len(patient_list))

    new_patient_list = []
    new_patient_ids = []
    counter = 0
    for patient in patient_list:
        if len(patient.infected_by) != 0:
            patient.idx_id = counter
            counter += 1
            new_patient_list.append(patient)
            new_patient_ids.append(patient.patient_id)
        else:
            add = False
            for patient_other in patient_list:
                if patient_other.infected_by == patient.patient_id:
                    add = True
            if add:
                patient.idx_id = counter
                counter += 1
                new_patient_list.append(patient)
                new_patient_ids.append(patient.patient_id)

    patient_list = new_patient_list
    patient_ids = new_patient_ids

    # print(len(patient_list))
    # label the patient ids with own labels (useful for matrix indexing)
    patient_id_to_idx_dic = {}
    idx_to_patient_id_dic = {}

    for idx, patient_id in enumerate(patient_ids):
        patient_id_to_idx_dic[patient_id] = idx
        idx_to_patient_id_dic[idx] = [patient_id]

    return patient_list, patient_ids, idx_to_patient_id_dic, patient_id_to_idx_dic

def get_patient_info_india(json_file):
    print(json_file)
    patient_list = []
    patient_ids = []
    counter = 0
    with open(json_file) as json_file:
        data = json.load(json_file)
        for i, p in enumerate(data['data']['rawPatientData']):
            patient = Patient()
            try:
                if len(p["contractedFrom"]) > 0:
                    patient.infected_by = p["contractedFrom"].split(',')[0]
                else:
                    patient.infected_by = "Z"
                
            except:
                patient.infected_by = "Z"
            if patient.infected_by[0] == 'P':
                counter += 1
                print(counter)
                print(patient.infected_by)
        

    exit()
    # We make the patient list sparser, to remove patients with no in or outgoing edges

    new_patient_list = []
    new_patient_ids = []
    counter = 0
    for patient in patient_list:
        if len(patient.infected_by) != 0:
            patient.idx_id = counter
            counter += 1
            new_patient_list.append(patient)
            new_patient_ids.append(patient.patient_id)
        else:
            add = False
            for patient_other in patient_list:
                if patient_other.infected_by == patient.patient_id:
                    add = True
            if add:
                patient.idx_id = counter
                counter += 1
                new_patient_list.append(patient)
                new_patient_ids.append(patient.patient_id)

    patient_list = new_patient_list
    patient_ids = new_patient_ids

    # print(len(patient_list))
    # label the patient ids with own labels (useful for matrix indexing)
    patient_id_to_idx_dic = {}
    idx_to_patient_id_dic = {}

    for idx, patient_id in enumerate(patient_ids):
        patient_id_to_idx_dic[patient_id] = idx
        idx_to_patient_id_dic[idx] = [patient_id]

    return patient_list, patient_ids, idx_to_patient_id_dic, patient_id_to_idx_dic




class Patient():
    def __init__(self, patient_id=None, idx_id=None, sex=None, infected_by=None, birthyear=None, date=None, city=None):
        self.city = city
        self.patient_id = patient_id
        self.idx_id = idx_id
        self.sex = sex
        self.infected_by=infected_by
        self.birthyear=birthyear
        self.date=date
        

if __name__ == "__main__":
    main()
