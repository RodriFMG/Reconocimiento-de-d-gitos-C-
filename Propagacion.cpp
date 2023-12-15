#include "NeuronalNetwork.h"

void NeuronalNetwork::forwardPropagation() {

    // Para la propagación para adelante se utilizó la siguiente fórmula:
    // z = w * cap + b || Para conseguir los scores
    // función de activación(z) || Para conseguir la capa próxima

    z1 = w1 * cap1 + b1;
    cap2 = (z1).unaryExpr([](double x) { return sigmoid(x); }) ;

    z2 = w2 * cap2 + b2;
    cap3 = (z2).unaryExpr([](double x) { return sigmoid(x); }) ;

    z3 = w3 * cap3 + b3;
    capfinal = softMax(z3) ;

}

void NeuronalNetwork::backPropagation() {
    forwardPropagation();


    // Fórmula matemática que he seguido para conseguir el gradiante entre capas:
    // Grad_FP * cap.transpose() = aL/aw = aL/az * az/aw || az/aw = transpuesta de la capa
    // Grad_FP * 1 = aL/ab = aL/az * az/ab || az/ab = 1

    // Forma de sacar la gradiante entre la penúltima capa y capa de salida (SOLO SI SE UTILIZA LA FA softmax)
    VectorXd Grad_FP3 = Gradiante_Funcion_Perdida(z3);
    w3 = w3 - tasa_aprendizaje * (Grad_FP3 * cap3.transpose());
    b3 = b3 - tasa_aprendizaje * (Grad_FP3 * 1);

    // Forma de sacar los gradiantes presentes entre la capa de entrada hasta la penúltima capa
    // (SOLO SI SE UTILIZA LA FA sigmoide)
    VectorXd Grad_FP2 = (w3.transpose() * Grad_FP3).cwiseProduct(z2.unaryExpr([](double x) { return derivada_sigmoid(x);}));
    w2 = w2 - tasa_aprendizaje * (Grad_FP2 * cap2.transpose());
    b2 = b2 - tasa_aprendizaje * (Grad_FP2 * 1);

    VectorXd Grad_FP1 = (w2.transpose() * Grad_FP2).cwiseProduct(z1.unaryExpr([](double x){ return derivada_sigmoid(x);}));
    w1 = w1 - tasa_aprendizaje * (Grad_FP1 * cap1.transpose());
    b1 = b1 - tasa_aprendizaje * (Grad_FP1 * 1);

    error = Funcion_Perdida(capfinal); // Cálculo del error.
}

