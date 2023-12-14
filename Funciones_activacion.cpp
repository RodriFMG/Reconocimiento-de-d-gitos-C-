//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

double NeuronalNetwork::sigmoid(double x) { //funcion de activacion (binaria - 2 capas solamente)
    return 1.0 / (1.0 + exp(-x));

    //se utiliza para conseguir el vector intermedio entre la ambas capas.
}

VectorXd  NeuronalNetwork::softMax(const VectorXd& x) { // funcion de activacion para la cada de salida
    VectorXd exp_xd = x.array().exp(); //para convertir los valores en una distribucion de probabilidad (multiclase - 3 o más capas)
    double sum_xd = exp_xd.sum();

    return exp_xd / sum_xd;
    //se utiliza para conseguir el vector intermedio entre la penultima capa y la capa final.
}

VectorXd NeuronalNetwork::Relu(const VectorXd& eigen) {
    return eigen.cwiseMax(0.0);
}