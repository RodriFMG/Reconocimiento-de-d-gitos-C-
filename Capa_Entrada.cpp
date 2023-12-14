//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

VectorXd NeuronalNetwork::neuronas_entrada(int divisor) { //almaceno las neuronas de entrada, para posteriormente
    //asignarsela a cap1.

    size_t tamanio_total{};

    for(auto [f,s] : um1){
        tamanio_total += static_cast<size_t>(round(s/divisor)); //Con esto calculo el tamaño del VectorXd
    }

    VectorXd vxd;

    vxd.resize(static_cast<Index>(tamanio_total)); //le asigno ese tamaño convirtiendolo implicitamente
    //a un Index (parte de la libreria Eigen que maneja el tipo de datos de indices de sus contenedores)

    size_t index = 0;
    for(auto [f,s] : um1){
        auto i = static_cast<size_t>(round(s/divisor));
        for(int j=0; j<i; ++j){
            vxd[static_cast<Index>(index)] = f; //En cada índice del VectorXd le almaceno el número que se repite
            //"i" veces.
            index++;
        }
    }

    return vxd;
}

