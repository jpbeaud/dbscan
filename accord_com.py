import pandas as pd 
import numpy as np
import time



chemin = "C:/Users/jpbea/anaconda3/envs/test1/data/accord_com/"
debut = time.time()
print("Début à", debut)

fichier_DH = pd.read_csv(chemin+"DH_0_3.csv", sep=";", dtype=str,skipinitialspace=True)

col_ref = ['cli_1']
references_DH = fichier_DH[col_ref]
#print("fichier_DH=", fichier_DH.head())

#fichier_DH = fichier_DH.drop('id_personne', axis='columns')
#fichier_DH.set_index('id_personne', inplace=True) # inplace=True --> on ne crée pas un nouveau DF

#____ reformatage des données ___________________________________
# id_personne	pcs	statut	csp	cosop

#fichier_DH["statut"] = fichier_DH["statut"].map({"AC":"01", "AU":"10", "CH":"15", "ET":"20","IN":"30","NA":"35", "RA":"40","RC":"45",
#                                                 "RE":"50","SP":"55"}, na_action=None)
fichier_DH = fichier_DH.replace("", np.nan)
fichier_DH = fichier_DH.fillna(0)

fichier_SIO = pd.read_csv(chemin+"g1ad_cl3.csv", sep=";", dtype=str,skipinitialspace=True)
col_ref = ['cli_1']
#___ tri des fichiers
fichier_DH.sort_values(by='cli_1')
fichier_SIO.sort_values(by='cli_1')
# transformation des données en numérique
fichier_SIO["dec_1"] = fichier_SIO["dec_1"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_2"] = fichier_SIO["dec_2"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_3"] = fichier_SIO["dec_3"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_4"] = fichier_SIO["dec_4"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_5"] = fichier_SIO["dec_5"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_6"] = fichier_SIO["dec_6"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_7"] = fichier_SIO["dec_7"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_8"] = fichier_SIO["dec_8"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_9"] = fichier_SIO["dec_9"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_10"] = fichier_SIO["dec_10"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_11"] = fichier_SIO["dec_11"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_12"] = fichier_SIO["dec_12"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_13"] = fichier_SIO["dec_13"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_14"] = fichier_SIO["dec_14"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_15"] = fichier_SIO["dec_15"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_16"] = fichier_SIO["dec_16"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_17"] = fichier_SIO["dec_17"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_18"] = fichier_SIO["dec_18"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_19"] = fichier_SIO["dec_19"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_20"] = fichier_SIO["dec_20"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_21"] = fichier_SIO["dec_21"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_22"] = fichier_SIO["dec_22"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_23"] = fichier_SIO["dec_23"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)
fichier_SIO["dec_24"] = fichier_SIO["dec_24"].map({"O":"1", "N":"2", "V":"1", "R":"2"}, na_action=None)


references_SIO= fichier_SIO[col_ref]
ref_cli = references_SIO.to_numpy()

fichier_SIO = fichier_SIO.replace("", np.nan)
fichier_SIO = fichier_SIO.fillna(0)
fichier_SIO = fichier_SIO.replace("3", "1")
fichier_SIO = fichier_SIO.replace("4", "2")



fichier_DH2 = fichier_DH
fichier_DH2 = fichier_DH2.drop('cli_1', axis='columns')
fichier_SIO = fichier_SIO.drop('cli_1',axis='columns')
# X = fichier_DH.to_numpy()
X = fichier_DH2
#X.set_index('id_personne', inplace=True) # inplace=True --> on ne crée pas un nouveau DF

# print(X.dtype)
X = X.apply(pd.to_numeric)

import matplotlib.pyplot as plt
# pltcatter(X[:, 3], X[:, 4])


#plt.scatter(x="pcs" , y="cosop", c = 'r', data=X)
#plt.show()

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# création du jeu de test

#X = StandardScaler().fit_transform(X)

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
min_groupe = 1
db = DBSCAN(eps=0.000000000000001,min_samples=min_groupe).fit(X)
labels_DH = db.labels_

n_clusters_ = len(set(labels_DH)) - (1 if -1 in labels_DH else 0)
n_noise_ = list(labels_DH).count(-1)

print("Estimated number of clusters DH: %d" % n_clusters_)
print("Estimated number of noise points DH: %d" % n_noise_)


Y = fichier_SIO
Y = Y.apply(pd.to_numeric)
db2 = DBSCAN(eps=0.00000000000001, min_samples=min_groupe).fit(Y)
labels_SIO = db2.labels_



n_clusters_ = len(set(labels_SIO)) - (1 if -1 in labels_SIO else 0)
n_noise_ = list(labels_SIO).count(-1)

print("Estimated number of clusters SIO: %d" % n_clusters_)
print("Estimated number of noise points SIO: %d" % n_noise_)
print("____________Résumé____________")
difference = labels_DH - labels_SIO
print("Données en écarts = ", difference)
print(labels_DH)
print(labels_SIO)
n_DH = len(labels_DH)
n_SIO=len(labels_SIO)
n_dif = len(difference)
print("Longueur dif = ", n_dif)
fin = False
i=-1

nb_ko = 0
while (i < n_dif - 1):
    i = i + 1
    if (difference[i] != 0):
        print("i=",i)
        print("différence enregistrement ", str(i), " DH=", labels_DH[i], "// SIO=",labels_SIO[i], "// ref=", ref_cli[i,:])
        #print("enr DH=", fichier_DH.loc[i,:])
        #print("enr SIO=",fichier_SIO.loc[i,:])
        nb_ko = nb_ko + 1
        X = np.delete(X,i,0)
        Y = np.delete(Y,i,0)
        ref_cli = np.delete(ref_cli,i,0)
        db = DBSCAN(eps=0.000000000000001,min_samples=min_groupe).fit(X)
        labels_DH = db.labels_
        db2 = DBSCAN(eps=0.00000000000001, min_samples=min_groupe).fit(Y)
        labels_SIO = db2.labels_
        difference = labels_DH - labels_SIO
        n_dif = n_dif - 1


lg = len(difference)
#nb_ano = 0
#for i in range(lg):
#    if difference[i] !=0:
#        nb_ano = nb_ano + 1
print("ANOMALIES = ", nb_ko)
#  3117773703

fin_exec = time.time()
temps = fin_exec - debut
print(f'Temps d\'exécution : {temps:.2}ms')