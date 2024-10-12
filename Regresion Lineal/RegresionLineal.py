import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model 


def grafica_regresion(data:pd.DataFrame, horas, prediccion): 
    plt.ylabel("Ingresos ($)") 
    plt.xlabel("Promedio de Horas trabajadas") 
    plt.scatter(x=data["horas"],y=data["ingreso"],color='pink')
    plt.scatter(horas, prediccion, color='green') 
    plt.plot(horas, prediccion, color='black')
    return plt.show()

def regresion_lineal(data:pd.DataFrame, entrada_horas): 
    regresion=linear_model.LinearRegression() 
    horas=data['horas'].values.reshape((-1,1)) 
    modelo=regresion.fit(horas, y=data['ingreso'])
    print("Interseccion (b): "+str(modelo.intercept_)) 
    print("Pendiente (m): "+str(modelo.coef_))
    print("Predicción: "+str(modelo.predict(entradas_horas)))
    grafica_regresion(data, entradas_horas, modelo.predict(entradas_horas))


data=pd.read_csv("ingreso.csv")  

entradas_horas=[[40],[45], [42],[39]] 
regresion_lineal(data, entradas_horas)



