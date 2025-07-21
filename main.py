import yfinance as yf
from sklearn.linear_model import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

symbol = "GOOGL"  # Simbolo a obtener info
data = yf.download(symbol, start="2018-01-01", end="2025-07-21")  # Lapso de tiempo de la info
##print(data) ##Verificar DataFrame

#####------- ESTO ES DATA PARA EL MODELO --------------- (DATA)
data = data[["Close"]]  # Obtenemos solo los precios de cierre del dataframe
data["SMA_10"] = (data["Close"].rolling(window=10).mean()) # Obtenemos la SMA de 10 periodos (Tendencia Corta)
data["SMA_50"] = (data["Close"].rolling(window=50).mean())  # Obtenemos la SMA de 50 periodos (Tendecia Media)
data["SMA_100"] = (data["Close"].rolling(window=100).mean())  # Obtenemos la SMA de 100 periodos (Tendecia Larga)
data["Return_diary"] = data["Close"].pct_change() ## Porcentaje de cambio intradiario
data["Volatilidad_10d"] = data["Return_diary"].rolling(window=3).std() #Volatilidad 3 periodos

#####---- Calculo RSI 10 periodos ---------- (DATA)
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=10).mean()
avg_loss = loss.rolling(window=10).mean()
rs = avg_gain / avg_loss
data["RSI_14"] = 100 - (100 / (1 + rs))


#### ---------------Funcion Objetivo (DATA)
data["Return_5d"] = data["Close"].shift(-5) / data["Close"] - 1  ##Funcion Objetivo
data["Return_5d_suavizado"] = data["Return_5d"].rolling(window=3).mean() ##Suavizamos la F objetivo


#### ---- Fin DATA limpieza datos vacios
data.dropna(inplace=True)  ##Limpiar datos


##### ----------------- Modelo
X = data[["Close", "SMA_10", "SMA_50", "SMA_100","Return_diary","Volatilidad_10d","RSI_14"]] #Caracteristicas
y = data["Return_5d_suavizado"] #Funcion Objetivo (Suavizada)




X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.20) #80% Datos entrenamiento - 20% Datos test



model = LinearRegression() #Regresion Lineal
model.fit(X_train, y_train) #Entrenamiento
y_pred = model.predict(X_test)#prediccion


plt.plot(y_test.values, label="Real")
plt.plot(y_pred, label="Predicho")
plt.title("Regresión de retorno a 5 días de "+ symbol)
plt.legend()
plt.grid(True)
plt.show()



num_dias = 30 # Define cuántos días quieres mostrar

# Selecciona los últimos 'num_dias' de los valores reales y predichos
# Como y_test es una Serie de Pandas, puedes usar .tail() o slicing
y_test_ultimos_10 = y_test.tail(num_dias)
y_pred_ultimos_10 = y_pred[-num_dias:] # y_pred es un array de NumPy, usa slicing

# Opcional: Si quieres que el eje X muestre las fechas reales en lugar de índices
# Puedes obtener los índices (fechas) correspondientes de X_test
fechas_ultimos_10 = X_test.tail(num_dias).index

# --- CÓDIGO DE LA GRÁFICA MODIFICADO ---

plt.figure(figsize=(10, 6)) # Opcional: para un gráfico más grande
plt.plot(fechas_ultimos_10, y_test_ultimos_10.values, label="Real")
plt.plot(fechas_ultimos_10, y_pred_ultimos_10, label="Predicho")

plt.title(f"Regresión de retorno de {symbol} (Últimos {num_dias} días)")
plt.xlabel("Fecha") # Cambia la etiqueta del eje X a "Fecha"
plt.ylabel("Retorno") # Añade etiqueta para el eje Y
plt.legend()
plt.grid(True) # Opcional: añade una cuadrícula para mejor lectura
plt.xticks(rotation=90) # Opcional: rota las etiquetas de fecha para que no se superpongan
plt.tight_layout() # Ajusta el diseño para evitar que las etiquetas se corten
plt.show()

