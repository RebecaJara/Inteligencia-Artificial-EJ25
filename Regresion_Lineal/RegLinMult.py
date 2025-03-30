# Imports necesarios
import numpy as np  
import pandas as pd  
import seaborn as sb 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm 
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score

#cargamos los datos de entrada
data = pd.read_csv("./articulos_ml.csv") 

# Filtramos los datos como antes
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]

# Creamos nueva variable combinando enlaces, comentarios e imágenes
suma = (filtered_data["# of Links"] + 
        filtered_data['# of comments'].fillna(0) + 
        filtered_data['# Images video'])

# Preparamos los datos para regresión múltiple
dataX2 = pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]  # Primera variable predictora
dataX2["suma"] = suma  # Segunda variable predictora (combinación de features)
XY_train = np.array(dataX2)  # Variables independientes (X1 y X2)
z_train = filtered_data['# Shares'].values  # Variable dependiente (Y)

# Creamos y entrenamos el modelo de regresión lineal múltiple
regr2 = linear_model.LinearRegression()

# Entrenamos el modelo, esta vez, con 2 dimensiones obtendremos 2 coeficientes, para graficar un plano
regr2.fit(XY_train, z_train)  # Ajuste del modelo

# Hacemos predicciones con el modelo entrenado
z_pred = regr2.predict(XY_train)

# Mostramos los resultados del modelo
# Los coeficientes
print("Coefficients: \n", regr2.coef_)
# Error cuadrático medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
# Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
print("Variance score: %.2f"% r2_score(z_train, z_pred))


# Visualización 3D del modelo
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))
 
# calculamos los valores del plano para los puntos x e y
nuevoX = (regr2.coef_[0] * xx)
nuevoY = (regr2.coef_[1] * yy)

# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regr2.intercept_)

# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')

# Graficamos en azul los puntos en 3D
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue', s=30, label='Datos reales')

# Graficamos en rojo las predicciones
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red', s=40, label='Predicciones')

# Configuramos los ejes y la vista
ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
ax.set_zlabel('Compartido en Redes')
ax.set_title('Regresión Lineal con Múltiples Variables')
ax.legend()

# Ajustamos la vista de la cámara
ax.view_init(elev=30., azim=65)

# Mostramos el gráfico
plt.show()

# Si quiero predecir cuántos "Shares" voy a obtener por un artículo con:
# 2000 palabras y con enlaces: 10, comentarios: 4, imagenes: 6
# según nuestro modelo, hacemos:
z_Dosmil = regr2.predict([[2000, 10+4+6]])
print("Predicción de shares para artículo nuevo:", int(z_Dosmil))
