#include "NeuronalNetwork.h"

void NeuronalNetwork::forwardPropagation(mutex& mtxs) {

    // Para la propagación para adelante se utilizó la siguiente fórmula:
    // z = w * cap + b || Para conseguir los scores
    // función de activación(z) || Para conseguir la capa próxima
    z1 = w1 * cap1 + b1;
    cap2 = z1.unaryExpr([](double x){ return NeuronalNetwork::Relu(x); });

    z2 = w2 * cap2 + b2;

    capfinal = NeuronalNetwork::softMax(z2);


}

void NeuronalNetwork::backPropagation(mutex& mtxs, Index index) {
    forwardPropagation(mtxs);


    // Fórmula matemática que he seguido para conseguir el gradiante entre capas:
    // Grad_FP * cap.transpose() = aL/aw = aL/az * az/aw || az/aw = transpuesta de la capa
    // Grad_FP * 1 = aL/ab = aL/az * az/ab || az/ab = 1

    // Forma de sacar la gradiante entre la penúltima capa y capa de salida (SOLO SI SE UTILIZA LA FA softmax)

    VectorXd Grad_FP2 = NeuronalNetwork::Gradiante_Funcion_Perdida(z2);
    w2 = w2 - tasa_aprendizaje * (Grad_FP2 * cap2.transpose());
    b2 = b2 - tasa_aprendizaje * (Grad_FP2 * 1);


    // Forma de sacar los gradiantes presentes entre la capa de entrada hasta la penúltima capa
    // (SOLO SI SE UTILIZA LA FA sigmoide)
    VectorXd Grad_FP1 = (w2.transpose() * Grad_FP2).cwiseProduct(z1.unaryExpr([](double x){ return NeuronalNetwork::derivada_Relu(x);}));

    w1 = w1 - tasa_aprendizaje * Grad_FP1 * cap1.transpose();
    b1 = b1 - tasa_aprendizaje * (Grad_FP1 * 1);


    /*
    if (i == max_iteraciones_entrenamiento) {
        // Normalizar al final del entrenamiento o después de un número específico de iteraciones
        w1.normalize();
        w2.normalize();
        b1.normalize();
        b2.normalize();
        i = 0;  // Reiniciar el contador de iteraciones
    }

    i++;

     */

    error = NeuronalNetwork::Funcion_Perdida(capfinal,index);
}

// VectorXd Grad_FP1 = (w2.transpose() * Grad_FP2).cwiseProduct(z1.unaryExpr([](double x){ return NeuronalNetwork::derivada_leakyRelu(x,0.01);}));