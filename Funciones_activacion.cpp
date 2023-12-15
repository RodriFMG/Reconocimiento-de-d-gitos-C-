//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

// Se utilizaron las siguientes funciones de activación:
// sigmoide: Para el forward-propagation entre la capa de entrada hasta la penúltima capa.
// softMax: Para el forward-propagation entre la penúltima capa y la capa de salida y para el proceso de entrenamiento.


// Como la función retorna un double, se tendrá que aplicar la FA elemento por elemento del vector de la capa.
double NeuronalNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

VectorXd  NeuronalNetwork::softMax(const VectorXd& x) {
    VectorXd exp_xd = x.array().exp();
    double sum_xd = exp_xd.sum();

    return exp_xd / sum_xd;
}

// No se utilizó, pero porsiacaso lo dejaré.
VectorXd NeuronalNetwork::Relu(const VectorXd& eigen) {
    return eigen.cwiseMax(0.0);
}