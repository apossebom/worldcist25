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

arquivo_argumentos_essenciais = url + "argumentos_essenciais.ob"



#Passo 1: Carregar a lista de argumentos essenciais
with open (arquivo_argumentos_essenciais, 'rb') as fp:
    base_argumentos_essenciais = pickle.load(fp)

print("Lista de argumentos carregada tem ", len(base_argumentos_essenciais), " elementos")

for i in range(len(base_argumentos_essenciais)):
    print("Argumento ", i, ":", base_argumentos_essenciais[i])

exit()






#Passo 2: Carregar o dataset binarizado
df_treino = pd.read_csv(dataset_binarizado_treino)




#Passo 3: Criar os algoritmos de machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
import random
import pandas as pd

# Função para gerar um caso de teste aleatório
def gerar_caso_teste():
    caso = {}

    # "Outlook" - apenas uma condição é True
    outlook_options = ["Overcast", "Rain", "Sunny"]
    chosen_outlook = random.choice(outlook_options)
    caso["Overcast"] = (chosen_outlook == "Overcast")
    caso["Rain"] = (chosen_outlook == "Rain")
    caso["Sunny"] = (chosen_outlook == "Sunny")

    # "Temperature" - apenas uma condição é True
    temperature_options = ["Cool", "Hot", "Mild"]
    chosen_temp = random.choice(temperature_options)
    caso["Cool"] = (chosen_temp == "Cool")
    caso["Hot"] = (chosen_temp == "Hot")
    caso["Mild"] = (chosen_temp == "Mild")

    # "Humidity" - apenas uma condição é True
    humidity_options = ["High", "Normal"]
    chosen_humidity = random.choice(humidity_options)
    caso["High"] = (chosen_humidity == "High")
    caso["Normal"] = (chosen_humidity == "Normal")

    # "Wind" - apenas uma condição é True
    wind_options = ["Strong", "Weak"]
    chosen_wind = random.choice(wind_options)
    caso["Strong"] = (chosen_wind == "Strong")
    caso["Weak"] = (chosen_wind == "Weak")

    return caso


# Geração de 10 casos de teste
casos_teste = [gerar_caso_teste() for _ in range(10)]

# Convertendo para DataFrame para visualização
df = pd.DataFrame(casos_teste)
#print("Casos de teste gerados:")
#print(df)



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


