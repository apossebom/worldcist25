import pandas as pd
import os 



url = os.path.dirname(os.path.abspath(__file__)) + "\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "PlayTennis.csv" 
dataset_original_treino_binarizado = url + "PlayTennis_binarizado.csv" 



coluna_target = "Play Tennis"




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



def binarizar_categorico(df_treino, nome_coluna, prefix=""):
        if prefix:
            coluna_treino = pd.get_dummies(df_treino[nome_coluna], drop_first=False, prefix=prefix)
        else:
            coluna_treino = pd.get_dummies(df_treino[nome_coluna], drop_first=False)
        
        return coluna_treino



dummies_outlook = binarizar_categorico(df_treino, 'Outlook')
dummies_temperature = binarizar_categorico(df_treino, 'Temperature')
dummies_humidity = binarizar_categorico(df_treino, 'Humidity')
dummies_wind = binarizar_categorico(df_treino, 'Wind')


#Gerar os novos datasets binarizados
df_final_treino = pd.concat([dummies_outlook, dummies_temperature, dummies_humidity, dummies_wind, y_treino], axis=1)

print(df_final_treino)



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


print(df_final_treino)

print("ORIGINAL: ")
print("Total de linhas: ", len(df_final_treino))
print("Total de colunas: ", df_final_treino.columns.size)





