# Importar librerias necesarias
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

# Leer archivo csv
dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")
print(dataframe.head())

print(dataframe.groupby('clase').size())

# Visualizacion de datos
dataframe.drop('clase', axis=1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase', height=4, vars=["duracion", "paginas","acciones","valor"], kind='reg')
plt.show()

# Creamos el Modelo de Regresión Logística
X = dataframe.drop('clase', axis=1)
y = dataframe['clase']
print(X.shape)

# Crear y entrenar modelo
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X,y)

predictions = model.predict(X)
print(predictions[:5])
print(model.score(X,y))

# Validación de nuestro modelo
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

name = 'LogisticRegression'
kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation,predictions))

# Reporte de Resultados del Modelo
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Clasificación de nuevos valores
X_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]})
model.predict(X_new)

# python RegLog.py