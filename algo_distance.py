import pandas as pd 
import numpy as np
from scipy.spatial import distance

def distance_deux(x,y):
    """ Retourne :
    - 0 si distance_euclidienne(x,y) = 0
    - 2 sinon"""
    dist = distance.euclidean(x,y)
    if dist == 0:
        return 0
    else:
        return 2

chemin = "C:/Users/jpbea/anaconda3/envs/test1/data/accord_com/"
print("Début")

fichier_DH = pd.read_csv(chemin+"fichier_DH_v3.csv", sep=";", dtype=str,skipinitialspace=True)

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

fichier_SIO = pd.read_csv(chemin+"g1ad_v3.csv", sep=";", dtype=str,skipinitialspace=True)
col_ref = ['cli_1']
fichier_DH.sort_values(by='cli_1')
fichier_SIO.sort_values(by='cli_1')
#print("fichier_SIO=", fichier_SIO.head())
references_SIO= fichier_SIO[col_ref]
ref_cli = references_SIO.to_numpy()
fichier_SIO = fichier_SIO.replace("", np.nan)
fichier_SIO = fichier_SIO.fillna(0)


fichier_DH2 = fichier_DH
fichier_DH2 = fichier_DH2.drop('cli_1', axis='columns')
fichier_SIO = fichier_SIO.drop('cli_1',axis='columns')
fichier_SIO = fichier_SIO.replace("3", "1")
fichier_SIO = fichier_SIO.replace("4", "2")
fichier_DH2 = fichier_DH2.apply(pd.to_numeric)
fichier_SIO = fichier_SIO.apply(pd.to_numeric)
X = fichier_DH2.to_numpy()
Y = fichier_SIO.to_numpy()
import time

#X = X.apply(pd.to_numeric)
N = len(X)
print("N=", N)
N = 1000



s = (N,N)
A = np.zeros(s)
B = np.zeros(s)

tps1 = time.time()

for i in range(N):
    for j in range(N):
      #  A[i,j] = distance.euclidean(X[i,:], X[j,:])
      A[i,j] = distance_deux(X[i,:], X[j,:])
tps2 = time.time()
print("Durée=", tps2 - tps1)
tps1 = time.time()

for i in range(N):
    for j in range(N):
      #  B[i,j] = distance.euclidean(Y[i,:], Y[j,:])
      B[i,j] = distance_deux(Y[i,:], Y[j,:])
tps2 = time.time()
print("Durée=", tps2 - tps1)
C = np.multiply(A,B) - A - B
print(len(C))
l1 = [i in range(N)]
l2 = [i in range(N)]
ano = 0
nb_ano=0
i_prev = -1
for i in range(N):
    for j in range(N):
        if (C[i,j] !=0):
            if (i != i_prev):
                nb_ano = nb_ano + 1
                i_prev = i
                print("Anomalie enr n° ",i, "Réf =",ref_cli[i,:])
            """if (ano < 100):
                print ("Anomalies pour ", i, j)
                #print(A[i,j], B[i,j])
                print("DH1=", X[i,:])
                print("DH2=", X[j,:])
                print("SIO1=", Y[i,:])
                print("SIO2=", Y[j,:])
                print("Réf. SIO : ",references_SIO.loc[i,:],"//", references_SIO.loc[j,:])
                print("Réf. DH : ",references_DH.loc[i,:], "//", references_DH.loc[j,:])
                ano = ano+1"""
print("N=", N)
print("Nombre d'anomalies = ", nb_ano)




