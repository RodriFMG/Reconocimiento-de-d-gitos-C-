//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

// Se aplica el factor xavier para conseguir matrices de pesos apropiados.
// (No es necesario si se trabaja con pocas capas, pero igual lo utilice)
void NeuronalNetwork::inicializacion_Xavier() {
    factor_xavier = sqrt(2.0/(static_cast<int>(cap1.size()) + n3));
}

void NeuronalNetwork::inicializar_capas(const Mat& img) {

    // Llamo a las funciones para conseguir el vector de neuronas para la capa de entrada.
    VectorXd guardar1 = MatToVector(img);
    VectorXdToUnorderedMap(guardar1);
    int divisor = minima_frecuencia_px();

    inicializacion_Xavier();

    // Se definen previamente las dimensiones de los vectores y matrices eigen, ya que, si no se define fallará el programa.
    // Es fundamental definir correctamente las dimensiones para poder realizar las operaciones entre matrices-vectores.
    // (suma, resta y producto son las que se utilizaron)

    // Dimension para el vector de neuronas: (n)
    // Dimension para la matriz de pesos: (n+1,n)
    // Dimension para el vector de sesgos: (n)
    // Dimension para el vector de scores: (n)

    cap1 = neuronas_entrada(divisor);
    w1 = MatrixXd::Random(n1, cap1.size()) * factor_xavier;
    z1 = VectorXd::Zero(cap1.size());

    b1 = VectorXd::Zero(n1);
    w2 = MatrixXd::Random(n2, n1) * factor_xavier;
    cap2 = VectorXd::Zero(n1);
    z2 = VectorXd::Zero(n1);

    b2 = VectorXd::Zero(n2);
    w3 = MatrixXd::Random(n3, n2) * factor_xavier;
    cap3 = VectorXd::Zero(n2);
    z3 = VectorXd::Zero(n2);

    b3 = VectorXd::Zero(n3);
    capfinal = VectorXd::Zero(n3);

    // VectorXd para el método de entrenamiento: One-Hot
    etiqueta_real = VectorXd::Zero(n3);
    etiqueta_real[indice] = 1;
}

// Constructor para definir las dimensiones, número de iteraciones (para el entrenamiento) y el índice del dígito
// (para el entrenamiento)

// La imagen se utiliza para la conversión de px a enteros, normalización y finalmente utilizarlo para
// asignar cada valor para cada neurona en la capa de entrada.
NeuronalNetwork::NeuronalNetwork(const Mat& img, int n1_1, int n2_1, int n3_1, int iter, int ind){
    n1 = n1_1;
    n2 = n2_1;
    n3 = n3_1;
    iteraciones = iter;
    indice = static_cast<Index>(ind);

    inicializar_capas(img);
}

NeuronalNetwork::~NeuronalNetwork() = default;