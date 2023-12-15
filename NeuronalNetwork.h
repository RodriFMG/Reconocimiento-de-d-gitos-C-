//
// Created by RODRIGO on 13/12/2023.
//

#ifndef PROYECTOPROGRAIII_NEURONALNETWORK_H
#define PROYECTOPROGRAIII_NEURONALNETWORK_H

#include <iostream>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <unordered_map>
#include <queue>
#include <vector>
#include <cmath>
#include <functional>

using namespace std;
using namespace cv;
using namespace Eigen;

class NeuronalNetwork {
private:
    int n1, n2, n3; // Dimensiones que tendrán los vectores y matrices eigen.

    MatrixXd w1, w2, w3; // Matrices eigen de pesos (matriz de los enlaces entre capas).

    VectorXd b1, b2, b3; // Vector eigen de sesgos (b1 es el vector de sesgos para cap2, b2 para cap3 y b3 para capfinal).

    VectorXd cap1, cap2, cap3, capfinal; // Vector eigen de neuronas por capa.

    VectorXd z1, z2, z3; // Vector eigen de los scores en la operación de: z = w * cap + b.

    VectorXd etiqueta_real; // Vector eigen one-hot para el entrenamiento (osea 1 para el valor correcto y 0 para los demás).

    // Se usaron para la normalización de px de la imagen para la capa de entrada.
    unordered_map<double,int> um1;
    priority_queue<double,vector<double>,greater<>> pq1;

    double factor_xavier{}; // Factor para la inicialización de las matrices de pesos.

    double tasa_aprendizaje = 0.001; // Tasa de aprendizaje colocada (para el entrenamiento).

    Index indice; // Índice del número de la imagen en el vector de resultados (utilizado para el vector one-hot).

    int iteraciones; // Número de veces que se repetirá el proceso. (para repetir la propagación para adelante y atrás).

    double error{}; // Guarda el costo final del proceso.

    vector<pair<int,double>> resultado; // Vector normal donde se guardarán los resultados al final del proceso.

public:

    // Orden para estudiar el código:

    // Normalización de la capa de entrada
    static VectorXd MatToVector(const Mat& cv);
    void VectorXdToUnorderedMap(const VectorXd& a);
    int minima_frecuencia_px();

    // Capa de entrada
    VectorXd neuronas_entrada(int divisor);

    // Red neuronal
    NeuronalNetwork(const Mat& img, int n1, int n2, int n3, int iteraciones, int indice);
    ~NeuronalNetwork();
    void inicializar_capas(const Mat& img);
    void inicializacion_Xavier();

    // Funciones de activación (FA)
    static double sigmoid(double x);
    static VectorXd softMax(const VectorXd& x);
    static VectorXd Relu(const VectorXd& eigen);

    // Derivadas de las funciones de activación
    static double derivada_sigmoid(double x);
    static VectorXd derivada_softMax(const VectorXd& x);

    // Descenso del gradiante
    [[nodiscard]] double Funcion_Perdida(const VectorXd& Nuevo) const ;
    [[nodiscard]] VectorXd Gradiante_Funcion_Perdida(const VectorXd& SoftMax);

    // Propagación
    void forwardPropagation();
    void backPropagation();

    // Resultados
    Index Prediccion();
    void Iteraciones();
    void resultados();
};

#endif //PROYECTOPROGRAIII_NEURONALNETWORK_H
