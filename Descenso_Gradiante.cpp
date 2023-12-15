//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

double NeuronalNetwork::Funcion_Perdida(const VectorXd& Funcion_activacion) const {

    // esta función de perdida = L = - sumatoria en j=1 hasta el #clases de (yj ln(^yj))
    // esto se reduce solamente a = L = -ln(^yk) = -ln(probabilidad de la clase correcta después de aplicar la FA softmax)

    VectorXd FA = softMax(Funcion_activacion);
    double prob = FA[indice];
    return -log(prob + 1e-10); //eñ 1e-10 es para evitar el log(0).
}

VectorXd NeuronalNetwork::Gradiante_Funcion_Perdida(const VectorXd& z) {
    //Para calcular el gradiante del error de perdida (Cross Entropy) utilizando la FA de softmax (no para otra FA):
    // Ósea: gradiante del costo / gradiante de z
    // = Vector de la predicción del modelo - etiqueta real || (Fórmula lo más simplificada posible)

    VectorXd softMaxs = softMax(z);
    return softMaxs - etiqueta_real;
}