//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

Index NeuronalNetwork::Prediccion() {
    Index maxIndex;
    capfinal.maxCoeff(&maxIndex); //saca el índice con mayor probabilidad.

    return maxIndex;
}

void NeuronalNetwork::Iteraciones(){
    for(int i=0; i<iteraciones; ++i){
        backPropagation();
    }

}

void NeuronalNetwork::resultados() {
    priority_queue<pair<int,double>, vector<pair<int,double>>,
            function<bool(const pair<int, double>& a, const pair<int,double>& b)>> digitos (
            [](const pair<int, double>& a, const pair<int,double>& b){
                return a.second < b.second;
            }
    );

    for(int i=0; i<10; ++i){
        digitos.emplace(i,capfinal[i]);
    }

    while(!empty(digitos)){
        resultado.push_back(digitos.top());
        digitos.pop();
    }

    double sum{};
    for(const auto&[f,s] : resultado){
        cout<<"Digito "<<f<<": "<<s<<endl;
        sum+=s;
    }
    cout<<"\n\n";

    cout<<"Probabilidad total: "<<sum<<endl;
    cout<<"Costo total: "<<error<<endl;
}