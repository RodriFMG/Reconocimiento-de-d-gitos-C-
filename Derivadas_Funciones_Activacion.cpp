//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

// Las derivadas de las funciones de activación nos servirán para el proceso del backpropagation.

double NeuronalNetwork::derivada_sigmoid(double x) {

    /* Hay 2 formas: Derivando simplemente el sigmoide
    return (exp(-x)/pow((1+exp(-x)),2));
    */

    // O... la menos compleja (por no usar el pow)
    return sigmoid(x) * (1.0 - sigmoid(x));
}

//// No se utilizó, pero porsiacaso lo deje.
[[maybe_unused]] VectorXd NeuronalNetwork::derivada_softMax(const Eigen::VectorXd &x) {

    /*  Esto es equivalente:
            VectorXd SM = softMax(x);
        VectorXd retorno= VectorXd::Zero(SM.size());

        for(Index i=0; i<SM.size();++i ){
           retorno[i] = SM[i] * (1 - SM[i]);
        }

       return retorno;
       */

    // Menos compleja al trabajar directamente con comandos de la librería eigen.


    return softMax(x).array() * (1 - softMax(x).array());
}

[[maybe_unused]] double NeuronalNetwork::derivada_Relu(double x) {
    return (x>0) ? 1 : 0;
}

double NeuronalNetwork::derivada_leakyRelu(double X, double alpha) {
    return (X > 0) ? 1 : alpha;
}

// return (X > 0) ? 1 : (abs(X)> -1e-25) ? -1e-25 : alpha / (abs(X) + 1e-8);