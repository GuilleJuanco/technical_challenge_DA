#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Cargar csv
df=pd.read_csv('data/measurements.csv')

#Limpieza y transformación

#Tirar columnas con menos de 100 no nulos
df.drop(['specials', 'refill liters', 'refill gas'], axis=1, inplace=True)

#Rellenar nulos de columna temp_inside con media
df['temp_inside'].fillna("69", inplace=True)

df['temp_inside'] = df['temp_inside'].str.replace(',', '.').astype(float)

media_ti = df['temp_inside'].mean()

df.loc[df['temp_inside'] == 69, 'temp_inside'] = media_ti

#Pasar columnas distance y consume a float

columnas = ['distance', 'consume']

for columna in columnas:
    df[columna] = df[columna].str.replace(',', '.').astype(float)


#Visualización

#Medias de consume, speed y distance para cada gas_type
media_por_tipo = df.groupby('gas_type')['consume', 'speed', 'distance'].mean()

#Dibujo
media_por_tipo.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Tipo de carburante')
plt.ylabel('Medias')
plt.title('Medias de Consumo, Velocidad, y Distancia por Tipo de carburante')
plt.legend()
plt.show()

#Valores
print(media_por_tipo)

#Modelo predictivo

#One-Hot Encoding de gas_type
df_e = pd.get_dummies(df.gas_type)

#Preparacion de datos
columnas_numericas = ['distance', 'speed', 'temp_inside', 'temp_outside', 'AC', 'rain', 'sun']
X = pd.concat([df_e, df[columnas_numericas]], axis=1)
y = df['consume']

#Divide datos para entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrena el modelo
model = LinearRegression()
model.fit(X_train, y_train)

#Predicciones
y_pred = model.predict(X_test)

#Evaluacion del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#Ejemplo1
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [1],
    'SP98': [0],
    'distance': [50],
    'speed': [60],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [1],
    'rain': [1],
    'sun': [0]
})

prediccion = model.predict(ejemplo)


print("Consumo esperado para E10 durante 50 millas a 60 m/h en lluvia:", prediccion)

#Ejemplo2
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [0],
    'SP98': [1],
    'distance': [50],
    'speed': [60],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [1],
    'rain': [1],
    'sun': [0]
})

prediccion = model.predict(ejemplo)

print("Consumo esperado para SP98 durante 50 millas a 60 m/h en lluvia:", prediccion)

#Ejemplo3
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [1],
    'SP98': [0],
    'distance': [50],
    'speed': [60],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [0]
})

prediccion = model.predict(ejemplo)

print("Consumo esperado para E10 durante 50 millas a 60 m/h en condiciones normales:", prediccion)

#Ejemplo4
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [0],
    'SP98': [1],
    'distance': [50],
    'speed': [60],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [0]
})

prediccion = model.predict(ejemplo)

print("Consumo esperado para SP98 durante 50 millas a 60 m/h en condiciones normales:", prediccion)

#Ejemplo5
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [1],
    'SP98': [0],
    'distance': [50],
    'speed': [10],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [1]
})


prediccion = model.predict(ejemplo)

print("Consumo esperado para E10 durante 50 millas a 10 m/h con sol:", prediccion)

#Ejemplo 6
# Parámetros 
ejemplo = pd.DataFrame({
    'E10': [0],
    'SP98': [1],
    'distance': [50],
    'speed': [10],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [1]
})

prediccion = model.predict(ejemplo)

print("Consumo esperado para SP98 durante 50 millas a 10 m/h con sol:", prediccion)

#Ejemplo7
#Parámetros
ejemplo = pd.DataFrame({
    'E10': [1],
    'SP98': [0],
    'distance': [50],
    'speed': [30],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [1]
})


prediccion = model.predict(ejemplo)

print("Consumo esperado para E10 durante 50 millas a 30 m/h con sol:", prediccion)

#Ejemplo8
# Parámetros 
ejemplo = pd.DataFrame({
    'E10': [0],
    'SP98': [1],
    'distance': [50],
    'speed': [30],
    'temp_inside': [20],
    'temp_outside': [15],
    'AC': [0],
    'rain': [0],
    'sun': [1]
})

prediccion = model.predict(ejemplo)

print("Consumo esperado para SP98 durante 50 millas a 30 m/h con sol:", prediccion)


'''Para todos los casos analizados,
y a diferencia de lo que se pudiese observar en el gráfico de medias
por tipo de carburante, la regresión lineal indica que SP98 consume menos
carburante que E10 para todos los casos posibles(condiciones climáticas,
distancia, velocidad...). El modelo de regresión lineal predice mayor consumo
para situaciones de lluvia y también para casos de alta velocidad y baja
velocidad, siendo más optimo mantener una velocidad moderada en cuanto a 
consumo. Este modelo no sería el más óptimo, de hecho esta bastante lejos
de ser óptimo como puede indicar su r-squared que solo explicaría el
9% de la varianza. El MRE también es muy alto, por lo que tendríamos
una tasa de error muy alta. Lo correcto sería proceder a realizar
un modelo random forest.'''