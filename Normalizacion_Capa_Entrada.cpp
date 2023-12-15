//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

// Lo que se busca es reducir el número de pixeles para la capa de entrada.

// Esta función nos permite realizar la conversión de pixeles a números enteros.
VectorXd NeuronalNetwork::MatToVector(const cv::Mat &cv) {
    Mat pe = cv.reshape(1,static_cast<int>(cv.total())); // Aplana la matriz de px con nxm a un (n*m)x1

    pe.convertTo(pe,CV_64FC1); // Realiza la conversión de px a números flotantes.

    return Map<VectorXd>(pe.ptr<double>(), static_cast<int>(pe.total())); // Convierto el Mat a un VectorXd.
}

// Esta función cuenta el número de veces que se repite cada valor del pixel en la imagen
void NeuronalNetwork::VectorXdToUnorderedMap(const Eigen::VectorXd &a) {
    for(auto ab : a){
        um1[ab]++; // Se utiliza un UnorderedMap para conseguir tanto el valor del pixel como la cantidad de
        // veces que se repite de la siguiente manera: "valor : cantidad"
    }
}

// Teniendo tanto el valor del pixel como número de veces que se repite, buscamos el pixel que menos
// presencia tiene en toda la imagen.
int NeuronalNetwork::minima_frecuencia_px() {
    for(auto [f,s] : um1){
        pq1.push(s); //Utilizando un priority_queue con greater, ordeno los pixeles de manera ascendente.
    }
    return static_cast<int>(pq1.top()); //retorno el pixel con menor presencia.
}

// Lo que se consigue con el pixel con menos frecuencia es que: dividimos la cantidad de veces que
// tiene la presencia el valor de un pixel en toda la imagen con respecto a la mínima frecuencia, ejemplo:

// px con menos frecuencia: 500
// px con mayor frecuencia: 20000

// cantidad de veces que se repetirá el px con menos frecuencia: 500/500 = 1
// cantidad de veces que se repetirá el px con mayor frecuencia: 20000/500 = 40
// donde significa que se repetirá 1 vez el valor del px con menos frecuencia y 40 veces el valor del px con mayor
// frecuencia en el vector de neuronas.

// En caso salga decimal la decimal, se redondeará aritméticamente.

// Así reduciendo significativamente la cantidad de neuronas en la capa de entrada sin perder "tanta" información.