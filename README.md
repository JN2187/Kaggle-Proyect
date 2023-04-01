# W7-Kaggle_competition

![Diamonds](https://live.staticflickr.com/8039/8010060159_775cc48e09_z.jpg)

# Readme

Este script de Python se utiliza para el preprocesamiento, modelado y visualización de datos. El script requiere tener instaladas las siguientes librerías: numpy, pandas, scipy, scikit-learn, imblearn, matplotlib, y seaborn.

# Preprocesamiento de datos

El script importa primero las librerías necesarias para el preprocesamiento de datos, incluyendo la imputación de datos (usando SimpleImputer, IterativeImputer, y KNNImputer), la estandarización de datos (usando MinMaxScaler, StandardScaler, y RobustScaler), y la codificación (usando LabelEncoder, OneHotEncoder, y OrdinalEncoder). El script también incluye métodos de remuestreo como el submuestreo y el sobremuestreo mediante RandomUnderSampler y RandomOverSampler.

# Visualización

El script importa matplotlib y seaborn para la visualización de datos. El script crea varios gráficos, incluyendo un histograma, un gráfico de distribución y un gráfico de caja, para visualizar la distribución y correlación de las variables.

# Modelado

El script importa varias librerías de aprendizaje automático, incluyendo LogisticRegression, LinearRegression, RandomForestClassifier, DecisionTreeClassifier, RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor, y GradientBoostingRegressor. El script también incluye métodos de validación cruzada y búsqueda en cuadrícula mediante cross_val_score y GridSearchCV. El script guarda el modelo entrenado usando pickle.

# Configuración

El script también incluye ajustes de configuración para advertencias y matplotlib para una mejor visualización.

# Resumen

El script importa los datos usando pandas. Lee sample_submission.csv, test.csv, y train.csv y crea una copia del conjunto de datos de entrenamiento (df). A continuación, el script realiza un análisis exploratorio de los datos comprobando los tipos de datos, el número de valores nulos y el número de puntos de datos duplicados.

El script crea varias visualizaciones para explorar la relación entre las variables y la variable objetivo. Crea un gráfico de histograma, un gráfico de distribución, un gráfico de caja y una matriz de correlación.

![Subplot](/images/Subplot.png)

Con este subplot visualizamos si hay orden o no en nuestras variables no numéricas, de cara a qué tipo de encoding utlizar más adelante. En este caso vemos que se aprecia cierto orden entre las variables, por lo que me decanto por un OrdinarEncoder.

Por último, compruebo cuál es el mejor modelo en base al MSE y al R2, decantándome por el Gradient Boosting ya que es el que mejor métricas tiene. Aunque aparentemente el Decision Tree I es el que tiene los mejores parámetros, tiene un poco ce overfiteo, ya que las diferecnias entre el train y el test son más notables en el MSE.

![Subplot2](/images/Subplot_modelo.png)

Nota: este README asume que el conjunto de datos se encuentra en una carpeta llamada "data" en el mismo directorio que el script de Python.
