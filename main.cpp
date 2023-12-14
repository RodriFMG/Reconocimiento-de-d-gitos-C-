#include <iostream>
#include "NeuralNetwork.h"

int main() {

    Mat img;

    img = imread(R"(C:\Users\RODRIGO\Pictures\Saved Pictures\ImagenNum1.png)", IMREAD_COLOR);
    int indice;
    cout<<"Coloca el indice del numero que buscaras [0,1,2,3,4,5,6,7,8,9]: ";cin>>indice;

    NeuronalNetwork NN(img, 12, 11, 10, 10000, indice);
    NN.Iteraciones();
    NN.resultados();
}
