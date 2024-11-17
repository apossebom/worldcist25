import pandas as pd
import os 
import numpy as np


url = os.path.dirname(os.path.abspath(__file__)) + "\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "seeds.csv" 
dataset_original_treino_binarizado = url + "seeds_binarizado.csv" 

coluna_target = "Type"




#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)

print("ORIGINAL: ")
print("Total de linhas: ", len(df_treino))
print("Total de colunas: ", df_treino.columns.size)




#Remover linhas com valore ausentes
df_treino = df_treino.dropna()



#Separar X e y
X_treino = df_treino.drop(coluna_target, axis=1)
y_treino = df_treino[coluna_target]



#Para Area #####################################
min_value = df_treino['Area'].min()
max_value = df_treino['Area'].max()
num_bins = 5
prefixo = "Area"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Area_label'] = pd.cut(df_treino['Area'], bins=bin_edges, labels=bins_labels)
dummies_Area = pd.get_dummies(df_treino['Area_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Perimeter #####################################
min_value = df_treino['Perimeter'].min()
max_value = df_treino['Perimeter'].max()
num_bins = 5
prefixo = "Perimeter"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Perimeter_label'] = pd.cut(df_treino['Perimeter'], bins=bin_edges, labels=bins_labels)
dummies_Perimeter = pd.get_dummies(df_treino['Perimeter_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Compactness #####################################
min_value = df_treino['Compactness'].min()
max_value = df_treino['Compactness'].max()
num_bins = 5
prefixo = "Compactness"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Compactness_label'] = pd.cut(df_treino['Compactness'], bins=bin_edges, labels=bins_labels)
dummies_Compactnessm = pd.get_dummies(df_treino['Compactness_label'], drop_first=False)
#print(dummies_sepalLengthCm)


#Para Kernel.Length #####################################
min_value = df_treino['Kernel.Length'].min()
max_value = df_treino['Kernel.Length'].max()
num_bins = 5
prefixo = "Kernel.Length"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Kernel_Length_label'] = pd.cut(df_treino['Kernel.Length'], bins=bin_edges, labels=bins_labels)
dummies_Kernel = pd.get_dummies(df_treino['Kernel_Length_label'], drop_first=False)
#print(dummies_sepalLengthCm)







#Para Kernel.Width #####################################
min_value = df_treino['Kernel.Width'].min()
max_value = df_treino['Kernel.Width'].max()
num_bins = 5
prefixo = "Kernel.Width"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Kernel.Width_label'] = pd.cut(df_treino['Kernel.Width'], bins=bin_edges, labels=bins_labels)
dummies_Kernel_Width = pd.get_dummies(df_treino['Kernel.Width_label'], drop_first=False)
#print(dummies_sepalLengthCm)





#Para Asymmetry.Coeff #####################################
min_value = df_treino['Asymmetry.Coeff'].min()
max_value = df_treino['Asymmetry.Coeff'].max()
num_bins = 5
prefixo = "Asymmetry.Coeff"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Asymmetry.Coeff_label'] = pd.cut(df_treino['Asymmetry.Coeff'], bins=bin_edges, labels=bins_labels)
dummies_Asymmetry_Coeff = pd.get_dummies(df_treino['Asymmetry.Coeff_label'], drop_first=False)
#print(dummies_sepalLengthCm)




#Para Kernel.Groove #####################################
min_value = df_treino['Kernel.Groove'].min()
max_value = df_treino['Kernel.Groove'].max()
num_bins = 5
prefixo = "Kernel.Groove"
bin_width = (max_value - min_value) / num_bins
bin_edges = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Kernel.Groove_label'] = pd.cut(df_treino['Kernel.Groove'], bins=bin_edges, labels=bins_labels)
dummies_Kernel_Groove = pd.get_dummies(df_treino['Kernel.Groove_label'], drop_first=False)
#print(dummies_sepalLengthCm)




df_final_treino = pd.concat([dummies_Area, dummies_Perimeter, dummies_Compactnessm, dummies_Kernel, dummies_Kernel_Width, dummies_Asymmetry_Coeff, dummies_Kernel_Groove, y_treino], axis=1)


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





