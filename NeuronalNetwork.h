//
// Created by RODRIGO on 13/12/2023.
//

#ifndef PROYECTOPROGRAIII_NEURONALNETWORK_H
#define PROYECTOPROGRAIII_NEURONALNETWORK_H

#include <iostream>
#include <Eigen/Dense>

#include <unordered_map>
#include <queue>
#include <vector>
#include <cmath>
#include <functional>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace chrono;
using namespace this_thread;

class NeuronalNetwork {
private:
    int n1, n2; // Dimensiones que tendrán los vectores y matrices eigen.

    int max_iteraciones_entrenamiento = 25, i{};

    MatrixXd w1, w2; // Matrices eigen de pesos (matriz de los enlaces entre capas).

    VectorXd b1, b2; // Vector eigen de sesgos (b1 es el vector de sesgos para cap2, b2 para cap3 y b3 para capfinal).

    VectorXd cap1, cap2, capfinal; // Vector eigen de neuronas por capa.

    VectorXd z1, z2; // Vector eigen de los scores en la operación de: z = w * cap + b.

    VectorXd etiqueta_real; // Vector eigen one-hot para el entrenamiento (osea 1 para el valor correcto y 0 para los demás).

    // Se usaron para la normalización de px de la imagen para la capa de entrada.
    unordered_map<double,int> um1;
    priority_queue<double,vector<double>,greater<>> pq1;

    double factor_xavier{}; // Factor para la inicialización de las matrices de pesos.

    //Re ajustar la tasa para asegurar al 100% de las predicciones.
    double tasa_aprendizaje = 0.00001; // Tasa de aprendizaje colocada (para el entrenamiento).

    Index indice{}; // Índice del número de la imagen en el vector de resultados (utilizado para el vector one-hot).

    vector<vector<pair<vector<double>, int>>> um; // Contenedor que almacenará por lotes el csv de datos MNIST (pair(px,número-índice))

    int iteraciones; // Número de veces que se repetirá el proceso. (para repetir la propagación para adelante y atrás).

    double error{}; // Guarda el costo final del proceso.

    string csv_Dato{};

    mutex mtx;

public:

    // Orden para estudiar el código:

    // Datos de entrada para el entrenamiento (datos MNIST en un csv)
    static void Concurrencia(vector<vector<pair<vector<double>, int>>>& um, ifstream& file, mutex& mtx, const int& filas, int hilos_pc);
    void Lectura_Concurrente(const string& fileName);

    // Red neuronal
    NeuronalNetwork(const string& Datos_Entrenamiento, const string& Dato, int n1, int n2, int iteraciones);
    ~NeuronalNetwork();
    void Entrenamiento();
    void Entrenamiento_Por_Lotes(vector<pair<vector<double>,int>>& Lotes, mutex& mtx);
    void Digitos_CSV();
    void inicializacion_Xavier();
    void Cambiando_Datos(VectorXd& Dato_Entrenamiento, const Index& index);

    // Funciones de activación (FA)
    static double sigmoid(double x);
    static VectorXd softMax(const VectorXd& vector);
    [[maybe_unused]] static double Relu( double eigen);
    static double leakyRelu(double x, double alpha);

    // Derivadas de las funciones de activación
    static double derivada_sigmoid(double x);
    [[maybe_unused]] static VectorXd derivada_softMax(const VectorXd& x);
    static double derivada_Relu(double x);
    static double derivada_leakyRelu(double x, double alpha);

    // Descenso del gradiante
    [[nodiscard]] double Funcion_Perdida(const VectorXd& Nuevo, Index index) const ;
    [[nodiscard]] VectorXd Gradiante_Funcion_Perdida(const VectorXd& SoftMax);

    // Propagación
    void forwardPropagation(mutex& mtx);
    void backPropagation(mutex& mtx, Index index);

    // Resultados


    VectorXd Sacar_Dato(const int& fila);
    void Prediccion(const int& fila);
    void resultados(const int& fila);

};

#endif //PROYECTOPROGRAIII_NEURONALNETWORK_H
