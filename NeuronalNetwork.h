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

using namespace std;
using namespace cv;
using namespace Eigen;

class NeuronalNetwork {
private:
    int n1, n2, n3;
    Index indice;
    int iteraciones;
    double error{};

    double factor_xavier{};
    double tasa_aprendizaje = 0.001;

    MatrixXd w1, w2, w3; //matriz de pesos: entrada - salida

    VectorXd b1, b2, b3; // Vector de sesgos de cada neurona en cada capa.

    VectorXd cap1, cap2, cap3; // neruonas de: capa de entrada - capa ocultas - capa final

    VectorXd z1, z2, z3; // enlaces entre la capa anterior con la capa posterior de: (z representa los scores)
    // capa input con capa de oculta - capa de entrada con capa oculta - capa oculta con la ultima capa

    //Es un paso para conseguir la matriz de pesos

    VectorXd capfinal; //Este será el vector del resultado esperado
    VectorXd etiqueta_real;

    unordered_map<double,int> um1;
    priority_queue<double,vector<double>,greater<>> pq1;
    unordered_map<int, double> digitos;

    static double sigmoid(double x);
    static VectorXd softMax(const VectorXd& x);
    static VectorXd Relu(const VectorXd& eigen);
    //static se pone cuando quiero llamar a la función sin usar ni un atributo privado de la clase
    //(no crea una instancia de la clase)

    static double derivada_sigmoid(double x);
    static VectorXd derivada_softMax(const VectorXd& x);



public:
    NeuronalNetwork(const Mat& img, int n1, int n2, int n3, int iteraciones, int indice);
    ~NeuronalNetwork();

    void forwardPropagation();
    void backPropagation();

    [[nodiscard]] double Funcion_Perdida(const VectorXd& Nuevo) const ;
    [[nodiscard]] VectorXd Gradiante_Funcion_Perdida(const VectorXd& SoftMax);

    static VectorXd MatToVector(const Mat& cv);
    void VectorXdToUnorderedMap(const VectorXd& a);
    int minima_frecuencia_px();
    VectorXd neuronas_entrada(int divisor);
    void inicializar_capas(const Mat& img);
    void inicializacion_Xavier();
    Index Prediccion();

    void Iteraciones();
    void resultados();
};


#endif //PROYECTOPROGRAIII_NEURONALNETWORK_H
