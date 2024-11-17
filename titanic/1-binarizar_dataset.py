import pandas as pd
import os 
import numpy as np



url = os.path.dirname(os.path.abspath(__file__)) + "\\"



############### DADOS DO DATASET ############################
dataset_original_treino = url + "titanic_train.csv" 
dataset_original_treino_binarizado = url + "titanic_train_binarizado.csv" 


dataset_original_teste = url + "titanic_test.csv" 
dataset_original_teste_binarizado = url + "titanic_test_binarizado.csv" 




coluna_target = "Survived"




#abrir dataset
df_treino = pd.read_csv(dataset_original_treino)
df_teste = pd.read_csv(dataset_original_teste)



print("ORIGINAL TREINO: ")
print("Total de linhas: ", len(df_treino))
print("Total de colunas: ", df_treino.columns.size)

print("ORIGINAL TESTE: ")
print("Total de linhas: ", len(df_teste))
print("Total de colunas: ", df_teste.columns.size)




#Remover linhas com valore ausentes
df_treino = df_treino.dropna()
df_teste = df_teste.dropna()



#Remover coluna Id caso exista
colunas_a_remover = ['PassengerId','Ticket', 'Name', 'Cabin']
df_treino = df_treino.drop(colunas_a_remover, axis=1)



#Remover linhas com valores ausentes
df_teste = df_teste.dropna()
df_teste = df_teste.drop_duplicates()



#Separar X e y
X_treino = df_treino.drop(coluna_target, axis=1)
y_treino = df_treino[coluna_target]

#X_teste = df_teste.drop(coluna_target, axis=1)  #teste não tem coluna target
#y_teste = df_teste[coluna_target]



#Para Pclass #####################################
dummies_pclass = pd.get_dummies(df_treino['Pclass'], prefix='Pclass')
#print(dummies_pclass)


#Para Sex #####################################
dummies_sex = pd.get_dummies(df_treino['Sex'], prefix='Sex')
#print(dummies_sex)


#Para Age #####################################
min_value = df_treino['Age'].min()
max_value = df_treino['Age'].max()
num_bins = 5
prefixo = "Age"
bin_width = (max_value - min_value) / num_bins
bin_edges_age = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels_age = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Age_label'] = pd.cut(df_treino['Age'], bins=bin_edges_age, labels=bins_labels_age)
dummies_age = pd.get_dummies(df_treino['Age_label'], drop_first=False)
#print(dummies_age)


#Para SibSp #####################################
dummies_Sibsp = pd.get_dummies(df_treino['SibSp'], prefix='SibSp')
#print(df['SibSp'].unique())
#print(dummies_Sibsp)


#Para Fare #####################################
min_value = df_treino['Fare'].min()
max_value = df_treino['Fare'].max()
num_bins = 6
prefixo = "Fare"
bin_width = (max_value - min_value) / num_bins
bin_edges_fare = np.linspace(min_value, max_value + bin_width, num_bins + 1)
bins_labels_fare = [prefixo+"_"+str(x) for x in range(num_bins)]
df_treino['Fare_label'] = pd.cut(df_treino['Fare'], bins=bin_edges_fare, labels=bins_labels_fare)
dummies_Fare = pd.get_dummies(df_treino['Fare_label'], drop_first=False)
#print(dummies_Fare)


#Para Parch #####################################
dummies_Parch = pd.get_dummies(df_treino['Parch'], prefix='Parch')
#print(dummies_Parch)


#Para Embarked #####################################
dummies_Embarked = pd.get_dummies(df_treino['Embarked'], prefix='Embarked')
#print(dummies_Embarked)




#Gerar os novos datasets binarizados

df_final_treino = pd.concat([dummies_pclass, dummies_sex, dummies_age, dummies_Sibsp, dummies_Fare, dummies_Parch, dummies_Embarked, y_treino], axis=1)

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

print("BINZARIZADO TREINO: ")
print("Total de linhas: ", len(df_final_treino))
print("Total de colunas: ", df_final_treino.columns.size)







##############TESTES
# Criação do dataset binarizado de teste (mantendo o código anterior até a criação de df_final_treino)

# Para Pclass #####################################
dummies_pclass_teste = pd.get_dummies(df_teste['Pclass'], prefix='Pclass')

# Para Sex #####################################
dummies_sex_teste = pd.get_dummies(df_teste['Sex'], prefix='Sex')

# Para Age #####################################
df_teste['Age_label'] = pd.cut(df_teste['Age'], bins=bin_edges_age, labels=bins_labels_age)
dummies_age_teste = pd.get_dummies(df_teste['Age_label'], drop_first=False)

# Para SibSp #####################################
dummies_Sibsp_teste = pd.get_dummies(df_teste['SibSp'], prefix='SibSp')

# Para Fare #####################################
df_teste['Fare_label'] = pd.cut(df_teste['Fare'], bins=bin_edges_fare, labels=bins_labels_fare)
dummies_Fare_teste = pd.get_dummies(df_teste['Fare_label'], drop_first=False)

# Para Parch #####################################
dummies_Parch_teste = pd.get_dummies(df_teste['Parch'], prefix='Parch')

# Para Embarked #####################################
dummies_Embarked_teste = pd.get_dummies(df_teste['Embarked'], prefix='Embarked')


# Salvar as colunas do treino para referência
colunas_treino = df_final_treino.columns

# Processo de binarização na base de teste
df_final_teste = pd.concat([dummies_pclass_teste, dummies_sex_teste, dummies_age_teste, dummies_Sibsp_teste, 
                            dummies_Fare_teste, dummies_Parch_teste, dummies_Embarked_teste], axis=1)
# Remover linhas duplicadas para garantir consistência
df_final_teste = df_final_teste.drop_duplicates()

# Garantir que as colunas da base de teste sejam iguais às do treino
for coluna in colunas_treino:
    if coluna not in df_final_teste.columns:
        df_final_teste[coluna] = 0  # Adiciona a coluna ausente com valor zero

# Reordena as colunas da base de teste para corresponder à base de treino
df_final_teste = df_final_teste[colunas_treino.drop(coluna_target)]  # Remove a coluna target para teste



# Remover linhas duplicadas para garantir consistência
print("TESTE: Tamanho atual: ", len(df_final_teste))
df_final_teste = df_final_teste.drop_duplicates()
print("TESTE: Tamanho depois de remover linhas duplicadas: ", len(df_final_teste))

# Salvar dataset binarizado da base de teste
df_final_teste.to_csv(dataset_original_teste_binarizado, index=False)

print(df_final_teste)
print("BINARIZADO TESTE:")
print("TESTE: Total de linhas: ", len(df_final_teste))
print("TESTE: Total de colunas: ", df_final_teste.columns.size)
