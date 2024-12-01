import pandas as pd 
import numpy as np 
import seaborn as sn 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm 
plt.rcParams['figure.figsize']=(16,9) 
plt.style.use('ggplot') 
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score 


class Articulos(): 
    path='../Archivos/articulos_ml.csv' 

    def LeerDatos(self): 
        data=pd.read_csv(self.path) 
        return data 
    
    def InformacionArchivo(self, data:pd.DataFrame): 
        print("Cantidad de Filas y Columnas: "+str(data.shape)) 
        print("Información: "+str(data.info())) 
        print("Nombre de columnas: "+str(data.columns)) 
        print("Descripción estadistica: "+str(data.describe()))




if (__name__=='__main__'): 
    articulo=Articulos() 
    data=articulo.LeerDatos()
    articulo.InformacionArchivo(data)
    regresion=data.drop(columns=['Title', 'url', 'Elapsed days']) 
    filtered_data=regresion[(regresion['Word count']<=3500) & (regresion['# Shares']<=80000)] 
    colores=['orange','blue'] 
    tamanio=[30,60] 
    word_count=filtered_data['Word count'].values 
    shared=filtered_data['# Shares'].values 

    asignar=[] 

    for index, row in filtered_data.iterrows(): 
        if row['Word count']>1808: 
            asignar.append(colores[0]) 
        else: 
            asignar.append(colores[1]) 
    
    ##plt.scatter(word_count, shared, c=asignar,s=tamanio[0] )
    ##plt.show()
    
    #Asignamos la variable Word count a nuestro modelo 
    
    dataX=filtered_data[['Word count']] 
    X_train=np.array(dataX)
    Y_train=filtered_data['# Shares'].values 
    
    regresion=linear_model.LinearRegression() 
    regresion.fit(X_train, Y_train) 

    y_pred=regresion.predict(X_train) 

    #Veamos los coeficientes obtenidos 
    print("Pendiente: "+str(regresion.coef_[0])) 
    print("Coordenadas independiente: "+str(regresion.intercept_))
    print("Error cuadratico medio %.2f" % mean_squared_error(Y_train,y_pred)) 
    print("Varianza score %.2f" % r2_score(Y_train, y_pred))

    nuevo_dato=regresion.predict([[2000]]) 
    print(int(nuevo_dato[0]))

    ##Vamos a mejorar nuestro modelo agregando una nueva variable 
    suma=(filtered_data['# of Links'].fillna(0) + filtered_data['# of comments'].fillna(0)+ 
          filtered_data['# Images video'].fillna(0)) 
    
    dataXY=pd.DataFrame() 
    dataXY['Word count']=filtered_data['Word count'] 
    dataXY['suma']=suma 
    
    XY_train=np.array(dataXY) 
    z_train=filtered_data['# Shares'].values 

    regresion.fit(XY_train,z_train) 

    z_predict=regresion.predict(XY_train) 

    print("Coeficiente: "+str(regresion.coef_)) 
    print("Termino independiente: "+str(regresion.intercept_)) 
    print("Error cuadratico medio %.2f" % mean_squared_error(z_train, z_predict)) 
    print("Varianza score: %.2f" % r2_score(z_train, z_predict))

    ##Graficaremos un plano en 3D 
    fig=plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    #ax=Axes3D(fig)

    ##creamos una malla sobre la cual graficaremos en el plano 
    xx, yy=np.meshgrid(np.linspace(0,3500, num=10), np.linspace(0,60, num=10)) 

    nuevo_xx=(regresion.coef_[0]*xx) 
    nuevo_yy=(regresion.coef_[1]*yy) 
    
    z=(nuevo_xx+nuevo_yy+regresion.intercept_) 

    ax.plot_surface(nuevo_xx, nuevo_yy, z, alpha=0.2, cmap='hot') 

    ax.scatter(XY_train[:,0], XY_train[:,1], z_train, c='blue', s=30) 
    ax.scatter(XY_train[:,0], XY_train[:,1], z_predict, c='red', s=50) 

    ax.view_init(elev=30, azim=60) 

    ax.set_xlabel('Cantidad de Palabras') 
    ax.set_ylabel('Cantidad de Link, comentarios,Imagenes') 
    ax.set_zlabel('Cantidad de compartidos') 
    ax.set_title('Regresión multiple variables')

    #plt.show()

    z_nuevo=regresion.predict([[2000,10+4+6]]) 
    print(int(z_nuevo[0]))

    
