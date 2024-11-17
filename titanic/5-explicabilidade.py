#Geração da lista de argumentos
import pandas as pd
import os 
from itertools import combinations
import pickle



url = os.path.dirname(os.path.abspath(__file__)) + "\\"


############### DADOS DO DATASET ############################
dataset_treino = url + "titanic_train.csv" 
dataset_binarizado_treino = url + "titanic_train_binarizado.csv" 

dataset_teste = url + "titanic_test.csv" 
dataset_binarizado_teste = url + "titanic_test_binarizado.csv" 

coluna_target = "Survived"

arquivo_argumentos_essenciais = url + "argumentos_essenciais.ob"



#Passo 1: Carregar a lista de argumentos essenciais
with open (arquivo_argumentos_essenciais, 'rb') as fp:
    base_argumentos_essenciais = pickle.load(fp)

print("Lista de argumentos carregada tem ", len(base_argumentos_essenciais), " elementos")





#Passo 2: Carregar o dataset binarizado
df_treino = pd.read_csv(dataset_binarizado_treino)
df_teste = pd.read_csv(dataset_binarizado_teste)



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



#Passo 4: Casos de teste não serão gerados. Existe o dataset de teste.



#Passo 5: Classificar os casos de teste com os algoritmos
fidelity_dt = 0
fidelity_knn = 0
fidelity_nb = 0

casos_teste = df_teste

for index, caso in df_teste.iterrows():

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


