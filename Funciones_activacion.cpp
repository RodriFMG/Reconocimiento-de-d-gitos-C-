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

VectorXd NeuronalNetwork::softMax(const VectorXd& x) {
    double max_x = x.maxCoeff();
    VectorXd exp_xd = VectorXd::Zero(x.size());

    for(int i=0; i<x.size(); ++i){
        exp_xd[i] = exp(x[i] - max_x);
    }

    double sum_exp_xd = exp_xd.sum();

    return exp_xd / ((sum_exp_xd==0) ? 1 : sum_exp_xd);
}

// No se utilizó, pero porsiacaso lo dejaré.
[[maybe_unused]] double NeuronalNetwork::Relu(double x) {
    /*
    VectorXd Relu = VectorXd::Zero(eigen.size());

    for(int i=0; i<eigen.size(); ++i){
        (eigen[i]>0) ? Relu[i] = eigen[i] : Relu[i] = 0;
    }

    return Relu;
     */

    //Otr forma

    const double max_val = 1e9;
    const double min_val = -1e9;
    x = max(min_val, min(max_val, x));

    return (x>=0) ? x : abs(0);
}

double NeuronalNetwork::leakyRelu(double x, double alpha) {
    const double max_val = 1e9;
    const double min_val = -1e9;
    x = max(min_val, min(max_val, x));

    return (x > 0) ? x : x * alpha;
}