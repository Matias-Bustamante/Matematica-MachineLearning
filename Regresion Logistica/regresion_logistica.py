
import pandas as pd 
import numpy as np 
from sklearn import linear_model 
from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sbn 

class LeerArchivo(): 

    url= '../Archivos/usuarios_win_mac_lin.csv'
    def LeerDatos(self)->pd.DataFrame:
        datos =pd.read_csv(self.url,sep=',') 
        return datos
    
    def InformacionEstadistica(self, data): 

        estadistica=data.describe() 
        return estadistica  
        
class ModeloRegresionLineal(): 

    def SeleccionVariables(self, data): 
        self.X=np.array(data.drop(columns='clase', axis=1, inplace=False)) 
        self.Y=np.array(data['clase'])  
        return (self.X, self.Y)
    
    def Entrenamiento(self): 
        self.model=linear_model.LogisticRegression() 
        return self.model.fit(self.X, self.Y) 
    
    def Predecir(self): 
        self.Entrenamiento()
         
        return self.model.predict(self.X) 
    
    def Precision(self): 
        return self.model.score(self.X, self.Y)
    
    def Validacion(self): 
        self.validation_size=0.20 
        self.seed=7 
        self.X_train, self.X_validation, self.Y_train, self.Y_validation=model_selection.train_test_split(self.X, self.Y, test_size=self.validation_size, random_state=self.seed) 
    
    def PrecisionMejorada(self): 
        
        self.name='Logistic Regression' 
        self.kfold=model_selection.KFold(n_splits=10, random_state=self.seed,shuffle=True) 
        self.resultado=model_selection.cross_val_score(self.model,self.X_train, self.Y_train, cv=self.kfold, scoring='accuracy' ) 
        self.mensaje="%s: %f: %f" % (self.name, self.resultado.mean(), self.resultado.std()) 
        print (self.mensaje)

    def NuevaPrediccion(self): 
        return self.model.predict(self.X_validation) 
    
    def NuevaPrecision(self): 

        print("Precision: "+str(accuracy_score(self.Y_validation, self.NuevaPrediccion())))





    

if __name__=='__main__': 
    Archivo=LeerArchivo() 
    data=Archivo.LeerDatos()
    info=Archivo.InformacionEstadistica(data)
    data.groupby('clase').size()
    ##data.drop(columns='clase', inplace=True, axis=1)
    ##data.hist()
    ##plt.show() 
    ##sbn.pairplot(data.dropna(), hue='clase', size=4, vars=["duracion", "paginas","acciones","valor"], kind='reg') 
    ##plt.show()
    RegresionLogistica=ModeloRegresionLineal() 
    resultado=RegresionLogistica.SeleccionVariables(data)
    print(RegresionLogistica.Predecir()[0:5])
    print("----Cálculo de Precision------") 
    print(RegresionLogistica.Precision())
    print("--------------------------------------------")
    RegresionLogistica.Validacion()
    RegresionLogistica.PrecisionMejorada()
    RegresionLogistica.NuevaPrecision()
