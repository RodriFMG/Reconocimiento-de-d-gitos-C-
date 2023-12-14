//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

VectorXd NeuronalNetwork::MatToVector(const cv::Mat &cv) {
    Mat pe = cv.reshape(1,static_cast<int>(cv.total()));

    pe.convertTo(pe,CV_64FC1);

    return Map<VectorXd>(pe.ptr<double>(), static_cast<int>(pe.total()));
}

void NeuronalNetwork::VectorXdToUnorderedMap(const Eigen::VectorXd &a) {
    for(auto ab : a){
        um1[ab]++;
    }
}

int NeuronalNetwork::minima_frecuencia_px() { //esta funcion es para conseguir el px con menos presencia en toda la imagen
    for(auto [f,s] : um1){ //la utilidad es para la normalizacion de neuronas en la capa de entrada.
        pq1.push(s);
    }
    return static_cast<int>(pq1.top());
}