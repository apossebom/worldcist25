#Geração da lista de argumentos
import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_original_treino = url + "iris.csv" 

dataset_binarizado_treino = url + "iris_binarizado.csv"

coluna_target = "Species"

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

"""
# Converter valores para booleanos (se necessário)
df_treino.iloc[:, :-1] = df_treino.iloc[:, :-1].applymap(lambda x: x == "True")

# Separar as features (atributos) e o rótulo (Species)
X = df_treino.drop(columns=['Species'])
y = df_treino['Species']

# Configurar o 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Realizar o 5-fold cross-validation e exibir cada teste
fold_number = 1
todos_testes = pd.DataFrame()
for train_index, test_index in kf.split(X):
    # Dividir o dataset em treino e teste com base nos índices do fold atual
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Resetar os índices para garantir que concatenaremos sem NaNs
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Exibir o fold atual, dados de treino e teste
    print(f"\nFold {fold_number}")
    #print("Dados de Treino:")
    #print(pd.concat([X_train, y_train], axis=1))
    print("Dados de Teste:")
    print(pd.concat([X_test, y_test], axis=1))
    
    #todos_testes.append(pd.concat([X_test, y_test], axis=1))
    dados_teste_atual = pd.concat([X_test, y_test], axis=1)
    todos_testes = pd.concat([todos_testes, dados_teste_atual], ignore_index=True)

    fold_number += 1

print("Todos os testes: ")
print(todos_testes)
"""

# Função para gerar um caso de teste aleatório para o dataset Iris binarizado
import random

def gerar_caso_teste():
    caso = {}

    # SepalLengthCm - apenas uma condição é True
    sepal_length_options = ["SepalLengthCm_0", "SepalLengthCm_1", "SepalLengthCm_2", "SepalLengthCm_3"]
    chosen_sepal_length = random.choice(sepal_length_options)
    for option in sepal_length_options:
        caso[option] = (option == chosen_sepal_length)

    # SepalWidthCm - apenas uma condição é True
    sepal_width_options = ["SepalWidthCm_0", "SepalWidthCm_1", "SepalWidthCm_2", "SepalWidthCm_3"]
    chosen_sepal_width = random.choice(sepal_width_options)
    for option in sepal_width_options:
        caso[option] = (option == chosen_sepal_width)

    # PetalLengthCm - apenas uma condição é True
    petal_length_options = ["PetalLengthCm_0", "PetalLengthCm_1", "PetalLengthCm_2", "PetalLengthCm_3"]
    chosen_petal_length = random.choice(petal_length_options)
    for option in petal_length_options:
        caso[option] = (option == chosen_petal_length)

    # PetalWidthCm - apenas uma condição é True
    petal_width_options = ["PetalWidthCm_0", "PetalWidthCm_1", "PetalWidthCm_2", "PetalWidthCm_3"]
    chosen_petal_width = random.choice(petal_width_options)
    for option in petal_width_options:
        caso[option] = (option == chosen_petal_width)

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
    #print(df_caso)
    for c in df_caso.columns:
        print(c, df_caso[c].values[0])
    
    # Classificar com cada algoritmo
    dt_r = dt.predict(pd.DataFrame([caso]))
    print("Decision Tree:", dt_r)
    knn_r = knn.predict(pd.DataFrame([caso]))
    print("K-Nearest Neighbors:", knn_r)
    nb_r = nb.predict(pd.DataFrame([caso]))
    print("Naive Bayes:", nb_r)

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


