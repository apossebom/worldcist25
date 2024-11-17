import pandas as pd
import os 
import numpy as np


url = os.path.dirname(os.path.abspath(__file__)) + "\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "iris.csv" 
dataset_original_treino_binarizado = url + "iris_binarizado.csv" 

coluna_target = "Species"




#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)

print("ORIGINAL: ")
print("Total de linhas: ", len(df_treino))
print("Total de colunas: ", df_treino.columns.size)




#Remover linhas com valore ausentes
df_treino = df_treino.drop("Id", axis=1)
df_treino = df_treino.dropna()



#Separar X e y
X_treino = df_treino.drop(coluna_target, axis=1)
y_treino = df_treino[coluna_target]



#Para SepalLengthCm #####################################
min_value = df_treino['SepalLengthCm'].min()
max_value = df_treino['SepalLengthCm'].max()
num_bins = 4
prefixo = "SepalLengthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['SepalLengthCm_label'] = pd.cut(df_treino['SepalLengthCm'], bins=bin_edges, labels=bins_labels)
dummies_sepalLengthCm = pd.get_dummies(df_treino['SepalLengthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para SepalWidthCm #####################################
min_value = df_treino['SepalWidthCm'].min()
max_value = df_treino['SepalWidthCm'].max()
num_bins = 4
prefixo = "SepalWidthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['SepalWidthCm_label'] = pd.cut(df_treino['SepalWidthCm'], bins=bin_edges, labels=bins_labels)
dummies_SepalWidthCm = pd.get_dummies(df_treino['SepalWidthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para PetalLengthCm #####################################
min_value = df_treino['PetalLengthCm'].min()
max_value = df_treino['PetalLengthCm'].max()
num_bins = 4
prefixo = "PetalLengthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['PetalLengthCm_label'] = pd.cut(df_treino['PetalLengthCm'], bins=bin_edges, labels=bins_labels)
dummies_PetalLengthCm = pd.get_dummies(df_treino['PetalLengthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)


#Para PetalWidthCm #####################################
min_value = df_treino['PetalWidthCm'].min()
max_value = df_treino['PetalWidthCm'].max()
num_bins = 4
prefixo = "PetalWidthCm"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['PetalWidthCm_label'] = pd.cut(df_treino['PetalWidthCm'], bins=bin_edges, labels=bins_labels)
dummies_PetalWidthCm = pd.get_dummies(df_treino['PetalWidthCm_label'], drop_first=False)
#print(dummies_sepalLengthCm)





df_final_treino = pd.concat([dummies_sepalLengthCm, dummies_SepalWidthCm, dummies_PetalLengthCm, dummies_PetalWidthCm, y_treino], axis=1)


print(df_final_treino.head())



#Remover linhas duplicadas para DF ficar consistente
print("TREINO: Tamanho atual: ", len(df_final_treino))
df_final_treino = df_final_treino.drop_duplicates()
print("TREINO: Tamanho depois de remover linhas duplicadas: ", len(df_final_treino))



colunas = df_final_treino.drop(coluna_target, axis=1)
duplicadas = df_final_treino.duplicated(subset=colunas, keep=False)
df_duplicates = df_final_treino[duplicadas]
df_final_treino = df_final_treino.drop(df_duplicates.index)
print("TREINO: Tamanho depois de remover duplicadas inconsistentes: ", len(df_final_treino))



#Salvar datasets binarizados
df_final_treino.to_csv(dataset_original_treino_binarizado, index=False)


print(df_final_treino.head())

print("BINARIZADO: ")
print("Total de linhas: ", len(df_final_treino))
print("Total de colunas: ", df_final_treino.columns.size)





