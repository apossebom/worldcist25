#Geração da lista de argumentos
import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_treino = url + "seeds.csv" 
dataset_binarizado_treino = url + "seeds_binarizado.csv" 

coluna_target = "Type"

arquivo_argumentos_essenciais = url + "argumentos_essenciais.ob"



#Passo 1: Carregar a lista de argumentos essenciais
with open (arquivo_argumentos_essenciais, 'rb') as fp:
    base_argumentos_essenciais = pickle.load(fp)

print("Lista de argumentos carregada tem ", len(base_argumentos_essenciais), " elementos")





#Passo 2: Carregar o dataset binarizado
df_treino = pd.read_csv(dataset_binarizado_treino)


resultados_unicos = df_treino[coluna_target].unique().tolist()



#Passo 3: Criar os algoritmos de machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#Treinar cada algoritmo com a base de treino
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
nb = GaussianNB()

dt.fit(df_treino.drop(columns=[coluna_target]), df_treino[coluna_target])
knn.fit(df_treino.drop(columns=[coluna_target]), df_treino[coluna_target])
nb.fit(df_treino.drop(columns=[coluna_target]), df_treino[coluna_target])



#Passo 4: gerar casos de teste aleatórios


# Função para gerar um caso de teste aleatório para o dataset Iris binarizado
import random

def gerar_caso_teste():
    caso = {}

    # Cada atributo principal tem cinco opções; escolhemos uma para ser `True`
    # "Area" - apenas uma condição é True
    area_options = ["Area_0", "Area_1", "Area_2", "Area_3", "Area_4"]
    chosen_area = random.choice(area_options)
    for option in area_options:
        caso[option] = (option == chosen_area)

    # "Perimeter" - apenas uma condição é True
    perimeter_options = ["Perimeter_0", "Perimeter_1", "Perimeter_2", "Perimeter_3", "Perimeter_4"]
    chosen_perimeter = random.choice(perimeter_options)
    for option in perimeter_options:
        caso[option] = (option == chosen_perimeter)

    # "Compactness" - apenas uma condição é True
    compactness_options = ["Compactness_0", "Compactness_1", "Compactness_2", "Compactness_3", "Compactness_4"]
    chosen_compactness = random.choice(compactness_options)
    for option in compactness_options:
        caso[option] = (option == chosen_compactness)

    # "Kernel.Length" - apenas uma condição é True
    kernel_length_options = ["Kernel.Length_0", "Kernel.Length_1", "Kernel.Length_2", "Kernel.Length_3", "Kernel.Length_4"]
    chosen_kernel_length = random.choice(kernel_length_options)
    for option in kernel_length_options:
        caso[option] = (option == chosen_kernel_length)

    # "Kernel.Width" - apenas uma condição é True
    kernel_width_options = ["Kernel.Width_0", "Kernel.Width_1", "Kernel.Width_2", "Kernel.Width_3", "Kernel.Width_4"]
    chosen_kernel_width = random.choice(kernel_width_options)
    for option in kernel_width_options:
        caso[option] = (option == chosen_kernel_width)

    # "Asymmetry.Coeff" - apenas uma condição é True
    asymmetry_coeff_options = ["Asymmetry.Coeff_0", "Asymmetry.Coeff_1", "Asymmetry.Coeff_2", "Asymmetry.Coeff_3", "Asymmetry.Coeff_4"]
    chosen_asymmetry_coeff = random.choice(asymmetry_coeff_options)
    for option in asymmetry_coeff_options:
        caso[option] = (option == chosen_asymmetry_coeff)

    # "Kernel.Groove" - apenas uma condição é True
    kernel_groove_options = ["Kernel.Groove_0", "Kernel.Groove_1", "Kernel.Groove_2", "Kernel.Groove_3", "Kernel.Groove_4"]
    chosen_kernel_groove = random.choice(kernel_groove_options)
    for option in kernel_groove_options:
        caso[option] = (option == chosen_kernel_groove)


    return caso


# Geração de 10 casos de teste
casos_teste = [gerar_caso_teste() for _ in range(10)]

# Convertendo para DataFrame para visualização
df = pd.DataFrame(casos_teste)
print("Casos de teste gerados:")
print(df)



#Passo 5: Classificar os casos de teste com os algoritmos
fidelity_dt = 0
fidelity_knn = 0
fidelity_nb = 0
for caso in casos_teste:
    print("\nCaso de teste:")
    df_caso = pd.DataFrame([caso])
    print(df_caso)
    
    # Classificar com cada algoritmo
    dt_r = dt.predict(pd.DataFrame([caso]))
    print("Decision Tree:", dt_r)
    knn_r = knn.predict(pd.DataFrame([caso]))
    print("K-Nearest Neighbors:", knn_r)
    nb_r = nb.predict(pd.DataFrame([caso]))
    print("Naive Bayes:", knn_r)

    #Encontrar explicações para cada classificação
    explicacoes = []
    for argumento in base_argumentos_essenciais:
        premissas = argumento["premissa"]
        conclusao = argumento["conclusao"]
        
        # Verificar se as premissas são verdadeiras no caso de teste
        premissas_verdadeiras = all([caso[p] for p in premissas])
        
        # Se todas as premissas são verdadeiras, a conclusão é válida
        if premissas_verdadeiras:
            explicacoes.append(argumento)
    
    print("Explicações encontradas:")
    for e in explicacoes:
        print(e)

    #Verificar a fidelity para cada algoritmo


    """
    if dt_r[0] == "Yes" and any([e["conclusao"] == "Yes" for e in explicacoes]):
        fidelity_dt += 1
    if knn_r[0] == "Yes" and any([e["conclusao"] == "Yes" for e in explicacoes]):
        fidelity_knn += 1
    if nb_r[0] == "Yes" and any([e["conclusao"] == "Yes" for e in explicacoes]):
        fidelity_nb += 1
    if dt_r[0] == "No" and any([e["conclusao"] == "No" for e in explicacoes]):
        fidelity_dt += 1
    if knn_r[0] == "No" and any([e["conclusao"] == "No" for e in explicacoes]):
        fidelity_knn += 1
    if nb_r[0] == "No" and any([e["conclusao"] == "No" for e in explicacoes]):
        fidelity_nb += 1
    """

    for species in resultados_unicos   :
        # Verifica se a resposta do modelo de árvore de decisão (dt_r) corresponde ao valor atual de species
        if dt_r[0] == species and any([e["conclusao"] == species for e in explicacoes]):
            fidelity_dt += 1
        
        # Verifica se a resposta do modelo KNN (knn_r) corresponde ao valor atual de species
        if knn_r[0] == species and any([e["conclusao"] == species for e in explicacoes]):
            fidelity_knn += 1
        
        # Verifica se a resposta do modelo Naive Bayes (nb_r) corresponde ao valor atual de species
        if nb_r[0] == species and any([e["conclusao"] == species for e in explicacoes]):
            fidelity_nb += 1
    
    print()


print("Acertos Decision Tree:", fidelity_dt)
print("Acertos K-Nearest Neighbors:", fidelity_knn)
print("Acertos Naive Bayes:", fidelity_nb)

f_dt = fidelity_dt / len(casos_teste)
f_knn = fidelity_knn / len(casos_teste)
f_nb = fidelity_nb / len(casos_teste)

print("Fidelity Decision Tree:", f_dt)
print("Fidelity K-Nearest Neighbors:", f_knn)
print("Fidelity Naive Bayes:", f_nb)


