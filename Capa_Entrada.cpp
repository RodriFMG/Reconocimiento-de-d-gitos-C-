//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

//En esta función le asigno todos los valores de los píxeles a un vector Eigen (VectorXd) lo cual representará
//la capa de entrada.

VectorXd NeuronalNetwork::neuronas_entrada(int divisor) {

    size_t tamanio_total{};

    for(auto [f,s] : um1){
        tamanio_total += static_cast<size_t>(round(s/divisor));
        //Con esto calculo el tamaño previo del vector Eigen (se necesita saber el espacio necesario
        // al crear el vector Eigen, ya que si no fallará).
    }

    VectorXd vxd;

    vxd.resize(static_cast<Index>(tamanio_total)); //Se le asigna el tamaño en memoria calculado.

    size_t index = 0;
    for(auto [f,s] : um1){
        auto i = static_cast<size_t>(round(s/divisor));
        for(int j=0; j<i; ++j){
            vxd[static_cast<Index>(index)] = f; //Le asigno cada valor del pixel al vector eigen
            index++;

            //el i vendría a ser la cantidad de veces que se repite ese pixel en toda la imagen
            //el f vendría siendo el valor del pixel
            //se utiliza el 2do for para repetir i veces el pixel con valor f
        }
    }

    return vxd; //Esto vendría siendo la capa de entrada utilizando los píxeles de la imagen.

    //Se utilizó el método feedforward para establecer la capa de entrada.
}

