import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sympy as sym 
from sklearn import linear_model 

def gradiente_descendiente_lineal(anio, salario, modelo, error): 
    
    ##Visualizando los datos y el modelo
    print("Intereseccion (b)= "+str(modelo.intercept_)) 
    print("Pendiente (m)= "+str(modelo.coef_[0]))
    print("Error cuadratico medio: %0.2f" % error_cuadratico)

    plt.figure(figsize=(8,6)) 
    plt.scatter(x=anio, y=salario, color='red', s=250, marker='o',label='Valor Verdadero')
    plt.scatter(x=anio, y=modelo.predict(anio), color='Green',s=250, marker='P', label='Valor Predicho')
    plt.plot(anio, modelo.predict(anio), linewidth=4, color='black', label='Modelo Lineal')

    plt.ylabel("Salario ($) ", size=16) 
    plt.xlabel("Años de Experiencia ", size=16)
    plt.legend(bbox_to_anchor=(1.3,0.5)) 
    plt.grid() 
    plt.box(False) 
    plt.show() 

def gradiente_dieferentes_modelo(anio, salario):
    anio=anio.reshape(3)
    #Creación de multiples pendientes para exploracion 
    pendientes=np.arange(2.5,1.6,-0.1) 

    #Vector para almacenar los diferentes errores del modelo 
    errores=np.array([]) 

    #Visualización del modelo 
    plt.figure(figsize=(8,6)) 

    for pendiente in pendientes: 
        #Error del modelo 
        error=((pendiente*anio-salario)**2).sum() 

        #Visualización del modelo para una pendiente dada 
        plt.plot(anio, pendiente*anio, linewidth=4, 
                 label='m: %0.2f | error: %0.2f' % (pendiente, error)) 

        errores=np.append(errores, error) 

    plt.scatter(anio, salario, color='green', s=250, 
                marker='o', label='Valor Verdadero') 
    plt.ylabel("Salario en ($) ", size=16) 
    plt.xlabel("Años de Experiencia ", size=16) 
    plt.legend(bbox_to_anchor=(1,0.5))  
    plt.grid()
    plt.box(False) 
    plt.show()  


##Regresion Lineal con Gradiente descendiente 

#variable independiente 
anio=np.array([[5],[6], [13]]) 

#variable objetivo 
salario=np.array([6.85, 16.83, 26.84]) 

#Regresion Lineal utilizando el metodo de Minimos Cuadrados
modelo=linear_model.LinearRegression().fit(anio.reshape(3,1), y=salario) 
error_cuadratico=((salario-modelo.predict(anio))**2).sum() 

#gradiente_descendiente_lineal(anio, salario, modelo, error_cuadratico)

gradiente_dieferentes_modelo(anio, salario=salario)