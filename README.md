# W7-Kaggle_competition

![Diamonds](https://live.staticflickr.com/8039/8010060159_775cc48e09_z.jpg)

# Readme

Este script de Python se utiliza para el preprocesamiento, modelado y visualización de datos. El script requiere tener instaladas las siguientes librerías: numpy, pandas, scipy, scikit-learn, imblearn, matplotlib, y seaborn.

# Preprocesamiento de datos

El script importa primero las librerías necesarias para el preprocesamiento de datos, incluyendo la imputación de datos (usando SimpleImputer, IterativeImputer, y KNNImputer), la estandarización de datos (usando MinMaxScaler, StandardScaler, y RobustScaler), y la codificación (usando LabelEncoder, OneHotEncoder, y OrdinalEncoder).

# Visualización

El script importa matplotlib y seaborn para la visualización de datos. El script crea varios gráficos, incluyendo un histograma, un gráfico de distribución y un gráfico de caja, para visualizar la distribución y correlación de las variables.

# Modelado

El script importa varias librerías de aprendizaje automático, incluyendo LogisticRegression, LinearRegression, RandomForestClassifier, DecisionTreeClassifier, RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor, y GradientBoostingRegressor. El script también incluye métodos de validación cruzada y búsqueda en cuadrícula mediante cross_val_score y GridSearchCV. El script guarda el modelo entrenado usando pickle.

# Configuración

El script también incluye ajustes de configuración para advertencias y matplotlib para una mejor visualización.


# Análisis exploratorio de datos y ajuste de modelos

Este proyecto tiene como objetivo realizar el análisis exploratorio de un conjunto de datos para visualizar la distribución de los datos, la correlación entre las variables, y estandarizar y codificar las variables para su posterior análisis de modelos de predicción.

## Análisis exploratorio

Se utiliza la función `value_counts()` para ver la frecuencia de los valores de la variable "price". Luego, se grafica la distribución de la variable "price" utilizando el gráfico `displot` de Seaborn. Se presentan los estadísticos descriptivos para las variables numéricas y categóricas con la función `describe()`. Posteriormente, se grafica la distribución para cada variable numérica utilizando el gráfico `histplot` de Seaborn.

En la siguiente sección, se grafica la relación entre las variables predictoras y la variable a predecir con la función `countplot` de Seaborn. Se verifica si las variables tienen un orden o no y se ve el ratio y la distribución de las variables no numéricas respecto al "price".

![Subplot](/images/Subplot.png)

Con este subplot visualizamos si hay orden o no en nuestras variables no numéricas, de cara a qué tipo de encoding utlizar más adelante. En este caso vemos que se aprecia cierto orden entre las variables, por lo que me decanto por un OrdinarEncoder.

Después, se visualizan los outliers con el gráfico `boxplot` de Seaborn y se crea una matriz de correlación con el gráfico `heatmap` de Seaborn.

## Preparación de los datos

Se estandarizan las variables numéricas con el método `RobustScaler()` de scikit-learn y se codifican las variables categóricas con el método `OrdinalEncoder()`. Los resultados se guardan en archivos .pkl.

## Ajuste de modelos

El objetivo es crear un modelo para predecir los precios de los diamantes, usando los diferentes métodos. El código primero carga los datos y los prepara, dividiéndolos en conjuntos de entrenamiento y prueba, y luego ajusta un modelo de regresión lineal a los datos de entrenamiento. Luego se hacen predicciones para los conjuntos de entrenamiento y prueba y se comparan con los valores reales. El código también genera visualizaciones para los valores reales y predichos, así como para los residuos del modelo. Se define una función para calcular varias métricas de evaluación del modelo, como el error absoluto medio, el error cuadrático medio y el coeficiente de determinación R². Finalmente, se ajusta un modelo de árbol de decisión y se realiza otra evaluación del modelo utilizando la misma función de métricas. El código también define un diccionario con los hiperparámetros para ajustar el modelo de árbol de decisión utilizando la búsqueda en cuadrícula.

## Evaluación del modelo

Se comparan los resultados de los modelos utilizando medidas de rendimiento como R² y MSE. Se utiliza la función de métricas definida previamente para calcular las métricas de evaluación del modelo. Decantándome por el Gradient Boosting ya que es el que mejor métricas tiene. Aunque aparentemente el Decision Tree I es el que tiene los mejores parámetros, tiene un poco de overfiteo, ya que las diferencias entre el train y el test son más notables en el MSE.

![Subplot2](/images/Subplot_modelo.png)


![Subplot2](/Result.png)

Nota: este README asume que el conjunto de datos se encuentra en una carpeta llamada "data" en el mismo directorio que el script de Python.
