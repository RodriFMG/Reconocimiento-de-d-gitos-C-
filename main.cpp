#include "NeuronalNetwork.h"

int main() {


    int opcion{};
    int fila{};

    NeuronalNetwork NN("mnist_train_short.csv","mnist_test_short.csv",256, 10, 1);

    do{
        cout<<"\t--Prediccion de digitos MNIST--"<<endl<<endl;

        cout<<"1. Predecir un digito"<<endl;
        cout<<"2. Salir"<<endl;
        cout<<endl<<endl;



        cout<<"Opcion: ";cin>>opcion;cout<<endl<<endl;

        switch (opcion) {
            case 1: cout<<"Elija la fila del digito que quiere predecir: ";cin>>fila;cout<<endl<<endl; NN.resultados(fila); break;
            case 2: break;
        }

    }while(opcion!=2);

    cout<<"\n\n";

}
