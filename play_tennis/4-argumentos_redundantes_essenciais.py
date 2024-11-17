#Geração da lista de argumentos
import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_original_treino = url + "PlayTennis.csv" 

dataset_binarizado_treino = url + "PlayTennis_binarizado.csv"

coluna_target = "Play Tennis"

arquivo_argumentos_possiveis= url + "argumentos_possiveis.ob"

arquivo_argumentos_consistentes = url + "argumentos_consistentes.ob"
arquivo_argumentos_inconsistentes = url + "argumentos_inconsistentes.ob"

arquivo_argumentos_redundantes = url + "argumentos_redundantes.ob"
arquivo_argumentos_essenciais = url + "argumentos_essenciais.ob"



#Passo 1: Carregar a lista de argumentos consistentes
with open (arquivo_argumentos_consistentes, 'rb') as fp:
    base_argumentos_consistentes = pickle.load(fp)
print("Lista carregada tem ", len(base_argumentos_consistentes), " elementos")   







def separar_argumentos(argumentos):
    """
    Função para separar uma lista de argumentos em essenciais e redundantes.

    Parâmetros:
    - argumentos: uma lista de dicionários, onde cada dicionário representa um
                  argumento com as chaves 'premissa', 'conclusao' e 'id'.
    
    Retorna:
    - Uma tupla (essenciais, redundantes) onde:
      - essenciais é uma lista de argumentos essenciais
      - redundantes é uma lista de argumentos redundantes
    """
    essenciais = []
    redundantes = []
    
    # Ordenar os argumentos por tamanho da premissa para garantir que as menores sejam verificadas primeiro
    argumentos = sorted(argumentos, key=lambda arg: len(arg["premissa"]))
    
    for i, arg_i in enumerate(argumentos):
        premissas_i = arg_i["premissa"]
        conclusao_i = arg_i["conclusao"]
        eh_redundante = False
        
        for arg_essencial in essenciais:
            # Se as premissas do argumento essencial estão contidas nas de arg_i e as conclusões são iguais
            if arg_essencial["premissa"].issubset(premissas_i) and arg_essencial["conclusao"] == conclusao_i:
                eh_redundante = True
                break
        
        if eh_redundante:
            redundantes.append(arg_i)
        else:
            essenciais.append(arg_i)
    
    return essenciais, redundantes



essenciais, redundantes = separar_argumentos(base_argumentos_consistentes)
print("Argumentos Essenciais:", len(essenciais))
for i in essenciais:
    print(i)


print("Argumentos Redundantes:", len(redundantes))
for i in redundantes:
    print(i)


with open(arquivo_argumentos_essenciais, 'wb') as fp:
    pickle.dump(essenciais, fp)
with open(arquivo_argumentos_redundantes, 'wb') as fp:
    pickle.dump(redundantes, fp)
