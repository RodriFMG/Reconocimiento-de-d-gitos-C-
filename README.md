En este repositorio se encuentra el enunciado 1 del proyecto de programación 3.  
Se implementó una red neuronal con la capacidad de reconocer dígitos mediante una infraestructura de feed forward con el método de entrenamiento one-hot.  
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

### Configuración de OpenCV

Configurar OpenCV puede ser más complejo y puede implicar ajustes en las variables de entorno y la instalación de una versión específica. Sigue estos pasos:

1. Descarga la versión 4.8.0 de OpenCV.
2. Verifica las instrucciones detalladas en [este video tutorial](https://www.youtube.com/watch?v=fjq8eTuHnMM&t=2s) y el [repositorio de ejemplo](https://github.com/Ethernel0/CmakeList-OpenCV).
3. Además, configura Clion con el compilador de Visual Studio Code según las instrucciones en [este video tutorial](https://www.youtube.com/watch?v=3ZinHm2HaQ8&t=783s).

Ten en cuenta que el video utiliza una versión anterior de OpenCV, pero podrías necesitar ajustar las variables de entorno para trabajar con "vc16". Después de modificar las variables de entorno, reinicia tu dispositivo.

### Configuraciones adicionales:
- Modifica las rutas de las imágenes ubicadas en el main según la ubicación real de las imágenes que planeas utilizar.

## Referencia:
- Canal de YT de [Ethernel Tech](https://www.youtube.com/@ethernel) (video y repostiorio)
