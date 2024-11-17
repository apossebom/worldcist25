import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_original_treino = url + "iris.csv" 
dataset_binarizado_treino = url + "iris_binarizado.csv" 

coluna_target = "Species"

arquivo_argumentos_possiveis= url + "argumentos_possiveis.ob"


#Passo 1: abrir dataset treino e teste
df_treino = pd.read_csv(dataset_binarizado_treino)


base_argumentos_possiveis = []
lista_premissas_argumentos_possiveis = []
id = 0


for index, linha in df_treino.iterrows():
    #print(linha.values)    #valores das colunas
    #print(linha.index)     #nomes das colunas

    X_teste = linha.drop(coluna_target)
    y_teste = linha[coluna_target]
    
    atributos_com_1 = [col for col in df_treino.columns if col != coluna_target and linha[col] == 1]
    print("Linha: ", atributos_com_1)


    for i in range(1, len(atributos_com_1)+1):
        combinacoes = combinations(atributos_com_1, i)
        for combinacao in combinacoes:
            temp = set(combinacao)            
            
            if temp not in lista_premissas_argumentos_possiveis:
                lista_premissas_argumentos_possiveis.append(temp)
                base_argumentos_possiveis.append({"premissa": temp, "conclusao": y_teste, "id": id})
                id += 1


for item in base_argumentos_possiveis:
    print(f"{item['id']}<{item['premissa']}, {item['conclusao']}>")
print("Total de Argumentos possíveis: ", len(base_argumentos_possiveis))



#Salvar lista de argumentos possíveis:
with open(arquivo_argumentos_possiveis, 'wb') as fp:
    pickle.dump(base_argumentos_possiveis, fp)

