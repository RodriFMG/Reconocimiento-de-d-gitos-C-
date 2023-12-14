//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

void NeuronalNetwork::inicializacion_Xavier() { // Para conseguir valores apropiados para mis matrices de pesos
    factor_xavier = sqrt(2.0/(static_cast<int>(cap1.size()) + n3));
}

void NeuronalNetwork::inicializar_capas(const Mat& img) {

    VectorXd guardar1 = MatToVector(img);
    VectorXdToUnorderedMap(guardar1);
    int divisor = minima_frecuencia_px();
    inicializacion_Xavier();

    //Cuando son pocas capas, puede usarse solo random. Pero decidi usar el factor xavier que se utiliza cuando son
    //muchas capas (estoy usando solo 2 capas ocultas).

    cap1 = neuronas_entrada(divisor);
    w1 = MatrixXd::Random(n1, cap1.size()) * factor_xavier;
    z1 = VectorXd::Zero(n1); //el factor xavier es para definir los pesos iniciales.

    //ver el problema del tamaño de los out1 y de los sesgos.

    b1 = VectorXd::Zero(n1);
    w2 = MatrixXd::Random(n2, n1) * factor_xavier;
    cap2 = VectorXd::Zero(n1); //en esta seccion defino un tamaño fijo para cada matriz o vector Eigen.
    z2 = VectorXd::Zero(n1);

    b2 = VectorXd::Zero(n2);
    w3 = MatrixXd::Random(n3, n2) * factor_xavier;
    cap3 = VectorXd::Zero(n2);
    z3 = VectorXd::Zero(n2);

    b3 = VectorXd::Zero(n3);
    capfinal = VectorXd::Zero(n3);

    //one - hot
    etiqueta_real = VectorXd::Zero(n3);
    etiqueta_real[indice] = 1;
}

NeuronalNetwork::NeuronalNetwork(const Mat& img, int n1_1, int n2_1, int n3_1, int iter, int ind){
    n1 = n1_1;
    n2 = n2_1;
    n3 = n3_1;
    iteraciones = iter;
    indice = static_cast<Index>(ind);

    inicializar_capas(img);
}

NeuronalNetwork::~NeuronalNetwork() = default;