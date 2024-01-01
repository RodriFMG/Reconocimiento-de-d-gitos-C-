En este repositorio se encuentra el enunciado 1 del proyecto de programación 3.  
Se implementó una red neuronal con la capacidad de reconocer dígitos MNIST mediante una infraestructura de feed forward con el método de entrenamiento one-hot (Supervisado).  
Se realizó en el lenguaje de programación C++ en el IDE (Entorno de Desarrollo Integrado) CLion.

# Configuración del Clion

## CMakeLists

Esta sección es crucial para el proyecto, ya que involucra la configuración de bibliotecas externas como Eigen y OpenCV.

### Configuración de Eigen

1. Descarga Eigen desde la [página oficial](https://gitlab.com/libeigen/eigen/-/releases/3.4.0) (versión 3.4.0).
2. Mueve la carpeta descargada de Eigen al directorio del proyecto, como se muestra en la siguiente estructura:

    ```
    Proyecto/
    ├── cmake-build-debug || En esta sección copias y pegas el contenido descargado de Eigen (toda la carpeta)
    
    ```

3. Modificar la ruta en el archivo `CMakeLists.txt` para que la referencia a Eigen esté correctamente configurada.
   
   ```
   cmake_minimum_required(VERSION 3.26)
   project(Reconocimiento-de-d-gitos-C-)
   set(CMAKE_CXX_STANDARD 23)
   
   set(EIGEN_DIR "[Ruta]/Reconocimiento-de-d-gitos-C-/cmake-build-debug/eigen-3.4.0")
   ```

# Csv de dígitos MNIST:
En el repositorio se compartio lo siguiente: 
- mnist_test_short.csv
- mnist_train_short.csv

Para que el programa pueda reconocer los csv, para el entranmiento y predicción, se deben colocar en: 
```
    Proyecto/
    ├── cmake-build-debug || En esta sección copias y pegas el contenido descargado de Eigen (toda la carpeta)

```

