# Reconocimiento Facial

## Funcionalidades

### 1. Personas registradas:

Es una sección informativa en la cual se pretende listar las personas de las cuales se ha entrenado el modelo.

### 2. Grabar video de nuevos rostros:

Permite capturar un video de 30 segundos de una persona la cual se debe identificar s¿con su nombre, se toma los frmaes mas claros en los que se identifique mejor el rostro y los guarda como Datos para entrenamiento.

Los datos o frames almacenados del rostro de una persona o varias personas se los toma para realizar el entrenamiento, el modelo resultantes se va a almacenar en un archivo XML, el cual va a poder ser usado en la sección 1. Reconocimiento Facial.

### 3. Reconocimiento Facial:

En base a un modelo almacenado, o despues de entrenar el modelo se lee el modelo entrnado y se realiza el reconocimiento de rostros.

## Instalations

* pip install flask
* pip install flask-cors
