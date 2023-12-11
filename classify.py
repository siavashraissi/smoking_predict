# Name: Siavash Raissi
# Course: CS 167
# Program Name: classify.py

import numpy as np 
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt



class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        pat_dists = []
        for pat in x_test:
            pat_dists.append(self.CalcDist(pat))
        
        np_pat_dists = np.array(pat_dists)
        x_NN_ind, y_NNs = self.findNN(np_pat_dists)

        y_pred = self.findMajority(x_NN_ind, y_NNs)

        return y_pred


    def CalcDist(self, pat):
        gene_dists = []
        for ref in self.x_train: # iterate through columns
            # iterate through genes 
            dist = np.linalg.norm(pat - ref)
            gene_dists.append(dist)

        return gene_dists

    def findNN(self, np_pat_dists):
        # list of the indices of the NNs in x_train as arrays
        x_NN_ind = []
        # list of the relating y_train data (arrays) for each index of x_NN_ind
        y_NNs = []

        for test_pat in np_pat_dists:
            # makes k-elt list of indices for the k-highest neighbors per patient
            ind = np.argpartition(test_pat, self.k)
            ind = ind[:self.k]
            # adds that list to the total list of patients 
            x_NN_ind.append(ind[:self.k])

            y_NN = self.y_train[ind[:self.k]]
            y_NNs.append(y_NN)

        # print(x_NN_ind)
        # print(y_NNs)

        return x_NN_ind, y_NNs

    def findMajority(self, x_NN_ind, y_NNs):
        majorities = []

        for pat in y_NNs:
            statuses, pos = np.unique(pat, return_inverse=True) 
            counts = np.bincount(pos)
            maj = counts.argmax()  

            majorities.append(statuses[maj])

        majorities_np = np.array(majorities)
        return majorities_np
            
        
def patientParse():
    df = pd.read_csv("GSE994-train.txt", sep="\t")

    y_train_df = df[-1:]
    y_train_np = y_train_df.to_numpy()

    x_train_df = df[:-1]
    x_train_np = x_train_df.to_numpy()
    x_train_np = x_train_np.astype(float)

    df2 = pd.read_csv("GSE994-test.txt", sep='\t')

    x_test_df = df2[:-1]
    x_test_np = x_test_df.to_numpy()
    x_test_np = x_test_np.astype(float)
    
    return x_train_np.T, y_train_np[0], x_test_np.T


def write_preds(x_train, y_train, x_test):
    # for 1NN
    NN1 = KNNClassifier(1, x_train, y_train)
    y_pred1 = NN1.predict(x_test)

    with open("Prob5-1NNoutput.txt", 'w') as f:
        patientnum = 31
        for pred in y_pred1:
            f.write('PATIENT' + str(patientnum))
            f.write('\t' + pred + '\n')
            patientnum += 1

    # for 3NN
    NN1 = KNNClassifier(3, x_train, y_train)
    y_pred3 = NN1.predict(x_test)

    with open("Prob5-3NNoutput.txt", 'w') as f:
        patientnum = 31
        for pred in y_pred3:
            f.write('PATIENT' + str(patientnum))
            f.write('\t' + pred + '\n')
            patientnum += 1
    
def foldAccuracy(x_train, y_train, k_vals):
    # y_train = y_train[0]
    fold_idxs = [(0,6), (6, 12), (12, 18), (18, 24), (24, 30)]

    y_pred_by_k = {}
    acc_by_k = {}

    for k in k_vals:
        fold_accuracies = []
        fold_preds = np.empty(0)
        for fold in fold_idxs:
            x_test = x_train[fold[0]:fold[1]]
            y_true = y_train[fold[0]:fold[1]]
            
            x_train_fold = np.concatenate((x_train[:fold[0]], x_train[fold[1]:]), axis=0)

            y_train_fold = np.concatenate((y_train[:fold[0]], y_train[fold[1]:]), axis=0)

            NNk = KNNClassifier(k, x_train_fold, y_train_fold)
            y_pred = NNk.predict(x_test)

            fold_preds = np.concatenate((fold_preds, y_pred))
            fold_accuracies.append(accuracy(y_true, y_pred))
        
        classifier_acc = sum(fold_accuracies) / len(fold_accuracies)
        acc_by_k[k] = classifier_acc
        y_pred_by_k[k] = fold_preds
            
    return y_pred_by_k, acc_by_k
        

def accuracy(y_true, y_pred):
    corr_count = 0
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            corr_count += 1
    
    acc = corr_count / len(y_true)
    return acc


def calc_TP_TN_FP_FN(y_true, y_pred):
    TP = TN = FP = FN = 0
    for i in range(0, len(y_true)):
        if y_true[i] == 'CurrentSmoker' and y_pred[i] == 'CurrentSmoker':
            TP += 1
        elif y_true[i] == 'NeverSmoker' and y_pred[i] == 'NeverSmoker':
            TN += 1
        elif y_true[i] == 'NeverSmoker' and y_pred[i] == 'CurrentSmoker':
            FP += 1
        elif y_true[i] == 'CurrentSmoker' and y_pred[i] == 'NeverSmoker':
            FN += 1
    return TP, TN, FP, FN


x_train, y_train, x_test = patientParse()
# write_preds(x_train, y_train, x_test)

k_vals = [1, 3, 5, 7, 11, 21, 23]
y_pred_by_k, acc_by_k = foldAccuracy(x_train, y_train, k_vals)

# graphing acc_by_k
# y_values = list(acc_by_k.values())
# plt.bar(range(len(acc_by_k)), y_values, tick_label=k_vals)
# plt.title('K Size versus Calculated Accuracy of Smoker Prediction', fontsize=10)
# plt.ylabel('Frequency of Correct Predictions', fontsize=10)
# plt.xlabel('K-values', fontsize=10)
# plt.savefig('knn_accuracies.png')


# FOR AGGLOMERATIVE CLUSTERING
clusters = AgglomerativeClustering(n_clusters=2, linkage='average').fit_predict(x_train)
clusters = np.where(clusters == 1, 'CurrentSmoker', 'NeverSmoker')
print(clusters)
# cluster_accuracy = accuracy(y_train, clusters)
# print(cluster_accuracy) # returns 0.5333333333333333


# COMPARING AGGLO CLUSTERING AND KNN CLASSIFIER
# NN_TP, NN_TN, NN_FP, NN_FN = calc_TP_TN_FP_FN(y_train, y_pred_by_k[5])
# clus_TP, clus_TN, clus_FP, clus_FN = calc_TP_TN_FP_FN(y_train, clusters)

# print(f'KNN Classifer (k=5):\n TP = {NN_TP}\n TN = {NN_TN}\n FP = {NN_FP}\n FN = {NN_FN}\n')
# print(f'AgglomerativeClustering:\n TP = {clus_TP}\n TN = {clus_TN}\n FP = {clus_FP}\n FN = {clus_FN}\n')
