import yfinance as yf

symbol = "AAPL"  # Simbolo a obtener info
data = yf.download(symbol, start="2018-01-01", end="2025-07-21")  # Lapso de tiempo de la info

##print(data) ##Verificar DataFrame

data = data[["Close"]]  # Obtenemos solo los precios de cierre del dataframe

##print (data) ##Verificar que solo sean los cierres

data["SMA_10"] = data["Close"].rolling(window=10).mean()  # Obtenemos la SMA de 10 periodos (Tendencia Corta)
print(data["SMA_10"].values)

data["SMA_50"] = data["Close"].rolling(window=50).mean()  # Obtenemos la SMA de 50 periodos (Tendecia Media)
print(data["SMA_50"].values)

data["SMA_100"] = data["Close"].rolling(window=100).mean()  # Obtenemos la SMA de 100 periodos (Tendecia Larga)
print(data["SMA_100"].values)

data["Return_5d"] = data["Close"].shift(-5) / data["Close"] - 1  ##Funcion Objetivo
print(data["Return_5d"].values)

data["Action"] = 0  # En el dataframe metemos columna llamada Action inciamos en 0
data.loc[data["Return_5d"] > 0.02, "Action"] = 1  # Si el retorno es superior al 2% el valor pasa a ser 1
data.loc[data["Return_5d"] < -0.02, "Action"] = -1  # Si la perdida es mayor al 2% el valor pasa a ser -1

data.dropna(inplace=True)  ##Limpiar datos
