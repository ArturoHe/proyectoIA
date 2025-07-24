import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler

symbol = "GOOGL"  # Simbolo a obtener info
data = yf.download(symbol, start="2018-01-01", end="2025-07-21")  # Lapso de tiempo de la info
##print(data) ##Verificar DataFrame

#####------- ESTO ES DATA PARA EL MODELO --------------- (DATA)
data["return"] = data['Close'].pct_change()

data["return_1d_ago"] = data["return"].shift(1)
data["return_2d_ago"] = data["return"].shift(2)
data["return_3d_ago"] = data["return"].shift(3)



#### ---------------Funcion Objetivo (DATA)
data["Return_5d"] = data["Close"].shift(-1) / data["Close"] - 1  ##Funcion Objetivo
data["Return_5d_suavizado"] = data["Return_5d"].rolling(window=3).mean()  ##Suavizamos la F objetivo

#### ---- Fin DATA limpieza datos vacios
data.dropna(inplace=True)  ##Limpiar datos

##### ----------------- Modelo
X = data[["return","return_1d_ago","return_2d_ago","return_3d_ago"]]  # Caracteristicas
y = data["Return_5d_suavizado"]  # Funcion Objetivo (Suavizada)

### ---------- Eliminacion de Valores atipicos
q_low = y.quantile(0.01)
q_high = y.quantile(0.99)
mask = (y > q_low) & (y < q_high)
X = X[mask]
y = y[mask]

#### ------------ Se escalan los X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

###---------- Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
                                                    test_size=0.20)  # 80% Datos entrenamiento - 20% Datos test

model = LinearRegression()  # Regresion Lineal
model.fit(X_train, y_train)  # Entrenamiento

###-------- Prediccion
y_pred = model.predict(X_test)  # prediccion

###----- Errores
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="neg_mean_squared_error")
print("MSE promedio (CV):", -scores.mean())


plt.plot(y_test.values, label="Real")
plt.plot(y_pred, label="Predicho")
plt.title("Regresión de retorno de " + symbol)
plt.legend()
plt.grid(True)
plt.show()

num_dias = 10  # Define cuántos días quieres mostrar

# Selecciona los últimos 'num_dias' de los valores reales y predichos
# Como y_test es una Serie de Pandas, puedes usar .tail() o slicing
y_test_ultimos_10 = y_test.tail(num_dias)
y_pred_ultimos_10 = y_pred[-num_dias:]  # y_pred es un array de NumPy, usa slicing

# Opcional: Si quieres que el eje X muestre las fechas reales en lugar de índices
# Puedes obtener los índices (fechas) correspondientes de X_test
fechas_ultimos_10 = X_test.tail(num_dias).index

# --- CÓDIGO DE LA GRÁFICA MODIFICADO ---

plt.figure(figsize=(10, 6))  # Opcional: para un gráfico más grande
plt.plot(fechas_ultimos_10, y_test_ultimos_10.values, label="Real")
plt.plot(fechas_ultimos_10, y_pred_ultimos_10, label="Predicho")

plt.title(f"Regresión de retorno de {symbol} (Últimos {num_dias} días)")
plt.xlabel("Fecha")  # Cambia la etiqueta del eje X a "Fecha"
plt.ylabel("Retorno")  # Añade etiqueta para el eje Y
plt.legend()
plt.grid(True)  # Opcional: añade una cuadrícula para mejor lectura
plt.xticks(rotation=45)  # Opcional: rota las etiquetas de fecha para que no se superpongan
plt.tight_layout()  # Ajusta el diseño para evitar que las etiquetas se corten
plt.show()
