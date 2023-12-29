cmake_minimum_required(VERSION 3.26)
project(Push_Nuevo)

set(CMAKE_CXX_STANDARD 23)

set(EIGEN_DIR "C:/Users/RODRIGO/Push_Nuevo/cmake-build-debug/eigen-3.4.0")

if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()


include_directories(${EIGEN_DIR})

add_executable(Push_Nuevo main.cpp
        Derivadas_Funciones_Activacion.cpp
        Descenso_Gradiante.cpp
        Entrenamiento_Entrada_Datos.cpp
        Funciones_activacion.cpp
        RedNeuronal.cpp
        Resultados.cpp
        Propagacion.cpp)

