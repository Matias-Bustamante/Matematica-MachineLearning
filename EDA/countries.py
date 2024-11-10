import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm 


class EDA_Countries(): 

    url='https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv' 

    def LeerDatos(self)->pd.DataFrame:
        data=pd.read_csv(self.url, sep=';')
        return data 
    
    def MostrarDatos(self, data, numero_registro)->pd.DataFrame: 
        print(data.head(numero_registro))
    
    def InformacionBasica(self, data):
        print("Cantidad de Filas y Columnas: "+str(data.shape))
        print("Nombre de columnas: "+str(data.columns))
        print("Informacion: "+str(data.info()))
        print("Descripción estadistica: "+str(data.describe()))

class Correlacion_Countries(): 

    #Se realiza una matriz de correlación para averiguar la relación entre variables
    def MatrizCorrelacion(self, countries): 
        self.data=countries[['alpha_3','area','geoname_id', 'numeric','population']] 
        self.corr=self.data.set_index('alpha_3').corr() 
        sm.graphics.plot_corr(self.corr, xnames=list(self.corr.columns)) 
        plt.show()

class Countries(): 
    url='https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv' 

    def LeerDatos(self)->pd.DataFrame: 
        self.data=pd.read_csv(self.url, sep=',') 
        return self.data 

    def MostrarPrimerosNRegistros(self,data, n)->pd.DataFrame: 
        print(data.head(n)) 
    
    def MostrarPoblacionEspaña(self, data:pd.DataFrame)->pd.DataFrame: 
        self.poblacion=data[data['country']=='Spain'] 
        return self.poblacion 
    
    def PoblacionArgentina(self, data:pd.DataFrame)->pd.DataFrame: 
        self.argentina=data[data['country']=='Argentina'] 
        return self.argentina 
    
    def CrearMatrizArgentinaEspaña(self, data:pd.DataFrame)->pd.DataFrame: 
        self.anio=data['year'].unique() 
        self.argentina=data[data['country']=='Argentina']['population'].values
        self.españa=data[data['country']=='Spain']['population'].values 

        matriz=pd.DataFrame({'Argentina':self.argentina, 
                             "España":self.españa},
                             index=self.anio)
        return matriz 
        
class Grafico(): 

    def diagramaDeBarra(self, data:pd.DataFrame)->pd.DataFrame: 
        data.drop(['country'],axis=1 )['population'].plot(kind='bar')
        plt.show()

class Anomalia(): 
    

    def find_anomalia(self, data:pd.DataFrame): 
        self.anomalia=[]
        desvio_estandar=data.std()
        desvio_estandar=desvio_estandar*2 
        media=data.mean() 
        limite_superior=media+desvio_estandar 
        limite_inferior=media-desvio_estandar 

        for index, row in  data.iterrows(): 
            if ((row.iloc[0]>limite_superior.iloc[0]) or (row.iloc[0]<limite_inferior.iloc[0])):
                self.anomalia.append(index) 

        return self.anomalia 

if (__name__=='__main__'):  
    
    EDA=EDA_Countries() 
    countries=EDA.LeerDatos() 
    EDA.MostrarDatos(countries,5)
    EDA.InformacionBasica(countries)
    Correlacion=Correlacion_Countries() 
    Correlacion.MatrizCorrelacion(countries=countries)

    Paises=Countries() 
    poblacion=Paises.LeerDatos() 
    Paises.MostrarPrimerosNRegistros(poblacion,5)
    pob_es=Paises.MostrarPoblacionEspaña(poblacion)
    argentina=Paises.PoblacionArgentina(poblacion)
    ##Paises.CrearMatrizArgentinaEspaña(poblacion).plot(kind='bar')
    ##plt.show()
    countries=countries.replace(np.nan,'',regex=True) 
    df_hispano=countries[countries['languages'].str.contains('es')]
    df_hispano.set_index('alpha_3')[['population','area']].plot(kind='bar', rot=65, figsize=(20,10)) 
    plt.show() 

    anomalia=Anomalia() 
    print(anomalia.find_anomalia(df_hispano.set_index('alpha_3')[['population']])) 
    df_hispano_corregido=df_hispano[df_hispano['alpha_3']!='USA'] 
    df_hispano_corregido=df_hispano_corregido[df_hispano_corregido['alpha_3']!='BRA']
    df_hispano_corregido.set_index('alpha_3')[['population','area']].plot(kind='bar', rot=65, figsize=(20,10)) 
    plt.show() 


    
