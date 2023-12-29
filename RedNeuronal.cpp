//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

// Se aplica el factor xavier para conseguir matrices de pesos apropiados.
// (No es necesario si se trabaja con pocas capas, pero igual lo utilice)
void NeuronalNetwork::inicializacion_Xavier() {
    factor_xavier = sqrt(2.0/(static_cast<int>(cap1.size()) + n2));
}

void NeuronalNetwork::Entrenamiento(){

    // Se definen previamente las dimensiones de los vectores y matrices eigen, ya que, si no se define fallará el programa.
    // Es fundamental definir correctamente las dimensiones para poder realizar las operaciones entre matrices-vectores.
    // (suma, resta y producto son las que se utilizaron)

    // Dimension para el vector de neuronas: (n)
    // Dimension para la matriz de pesos: (n+1,n)
    // Dimension para el vector de sesgos: (n)
    // Dimension para el vector de scores: (n)

    //cap1 tiene 784 de tamaño.
    cap1 = VectorXd::Zero(784);

    inicializacion_Xavier();

    w1 = MatrixXd::Random(n1, 784) * factor_xavier;
    z1 = VectorXd::Zero(n1);
    b1 = VectorXd::Zero(n1);


    w2 = MatrixXd::Random(n2, n1) * factor_xavier;
    cap2 = VectorXd::Zero(n1);
    z2 = VectorXd::Zero(n2);
    b2 = VectorXd::Zero(n2);

    capfinal = VectorXd::Zero(n2);

    // VectorXd para el método de entrenamiento: One-Hot
    etiqueta_real = VectorXd::Zero(n2);
}

void NeuronalNetwork::Cambiando_Datos(VectorXd& Dato_Entrenamiento, const Index& index){
    cap1 = Map<VectorXd>(Dato_Entrenamiento.data(),Dato_Entrenamiento.size());
    etiqueta_real[index] = 1;
}

//puede que falle...
void NeuronalNetwork::Entrenamiento_Por_Lotes(vector<pair<vector<double>,int>>& Lotes, mutex& mtx){

    for(auto& [px,index] : Lotes){
        VectorXd Digito = Map<VectorXd>(&px[0], static_cast<Index>(px.size()));
        auto Indice = static_cast<Index>(index);

        Cambiando_Datos(Digito,Indice);
        backPropagation(mtx, Indice);
        etiqueta_real = VectorXd::Zero(n2);

    }

}

void NeuronalNetwork::Digitos_CSV() {

    Entrenamiento();

    for(int i=0; i < iteraciones; ++i){
        for(auto pairs : um){
            Entrenamiento_Por_Lotes(pairs,mtx);
        }
    }

}

// Constructor para definir las dimensiones, número de iteraciones (para el entrenamiento) y el índice del dígito
// (para el entrenamiento)

// La imagen se utiliza para la conversión de px a enteros, normalización y finalmente utilizarlo para
// asignar cada valor para cada neurona en la capa de entrada.
NeuronalNetwork::NeuronalNetwork(const string& Datos_Entrenamiento, const string& Dato, int n1_1, int n2_1,
                                 int iter){
    n1 = n1_1;
    n2 = n2_1;
    iteraciones = iter;
    csv_Dato = Dato;

    Lectura_Concurrente(Datos_Entrenamiento);
    Digitos_CSV();

}

NeuronalNetwork::~NeuronalNetwork() = default;