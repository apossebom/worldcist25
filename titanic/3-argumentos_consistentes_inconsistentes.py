import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_original_treino = url + "titanic_train.csv" 

dataset_binarizado_treino = url + "titanic_train_binarizado.csv" 

coluna_target = "Survived"

arquivo_argumentos_possiveis= url + "argumentos_possiveis.ob"

arquivo_argumentos_consistentes = url + "argumentos_consistentes.ob"
arquivo_argumentos_inconsistentes = url + "argumentos_inconsistentes.ob"



#Passo 1: abrir dataset treino
df_treino = pd.read_csv(dataset_binarizado_treino)



#Passo 2: Carregar a lista de argumentos possíveis
with open (arquivo_argumentos_possiveis, 'rb') as fp:
    base_argumentos_possiveis = pickle.load(fp)
print("Lista carregada tem ", len(base_argumentos_possiveis), " elementos")



#Passo 3: Percorrer cada combinação para verificar se é um argumento consistente ou inconsistente
total = 0
listaArgumentosValidos = []
listaArgumentosMultiplasRespostas = []

for item in base_argumentos_possiveis:
    #if(total % 1000 == 0):
    #    print("Processando ", total, " de ", len(base_argumentos_possiveis))
    print("Processando ", total, " de ", len(base_argumentos_possiveis)) #//////////////

    condicao = (df_treino[list(item["premissa"])] == True).all(axis=1)
    df_filtrado = df_treino.loc[condicao]
    valores_unicos = df_filtrado[coluna_target].unique()

    if(len(valores_unicos) == 1 and valores_unicos[0] == item["conclusao"]):
        listaArgumentosValidos.append(item)
        print(f"a {valores_unicos} ", end="")
    elif(len(valores_unicos) == 1 and valores_unicos[0] != item["conclusao"]):
        item["conclusao"] = valores_unicos[0]
        listaArgumentosMultiplasRespostas.append(item)
        print(f"b {valores_unicos} ", end="")
        #input("Pressione uma tecla")
    else:
        item["conclusao"] = ""
        listaArgumentosMultiplasRespostas.append(item)
        print(f"c {valores_unicos} ", end="")
    print(item)
    
    total += 1

print("Total de argumentos consistentes: ", len(listaArgumentosValidos))
print("Total de argumentos inconsistentes: ", len(listaArgumentosMultiplasRespostas))

print() 

print("Lista de argumentos inconsistentes")
for it in listaArgumentosMultiplasRespostas:
    print(it)
print("\nLista de argumentos consistentes")
for it in listaArgumentosValidos:
    print(it)


print("Total de argumentos inconsistentes: ", len(listaArgumentosMultiplasRespostas))
print("Total de argumentos consistentes: ", len(listaArgumentosValidos))


with open(arquivo_argumentos_consistentes, 'wb') as fp:
    pickle.dump(listaArgumentosValidos, fp)
with open(arquivo_argumentos_inconsistentes, 'wb') as fp:
    pickle.dump(listaArgumentosMultiplasRespostas, fp)
