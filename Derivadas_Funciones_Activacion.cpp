//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

double NeuronalNetwork::derivada_sigmoid(double x) {
    /* Hay 2 formas: Derivando simplemente el sigmoide
    return (exp(-x)/pow((1+exp(-x)),2));
    */

    // O... (más optima por no usar el pow())
    return sigmoid(x) * (1.0 - sigmoid(x));
}

VectorXd NeuronalNetwork::derivada_softMax(const Eigen::VectorXd &x) {
    return softMax(x).array() * (1 - softMax(x).array());

    /*  Esto es equivalente:
            VectorXd SM = softMax(x);
        VectorXd retorno= VectorXd::Zero(SM.size());

        for(Index i=0; i<SM.size();++i ){
           retorno[i] = SM[i] * (1 - SM[i]);
        }

        return retorno;
     */
}