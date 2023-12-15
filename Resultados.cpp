//
// Created by RODRIGO on 14/12/2023.
//

#include "NeuronalNetwork.h"

//Esta función sirve para obtener el índice con mayor valor del vector de la capa final.
Index NeuronalNetwork::Prediccion() {
    Index maxIndex;
    capfinal.maxCoeff(&maxIndex); // Obtiene el índice del mayor valor.
    return maxIndex;
}

// Función para realizar el proceso la cantidad de veces definidas en el constructor.
void NeuronalNetwork::Iteraciones(){
    for(int i=0; i<iteraciones; ++i){
        backPropagation();
    }
}

// Función para obtener los resultados (Se utiliza después de la función de "Iteraciones", para obtener los
// resultados finales después de todo el proceso)
void NeuronalNetwork::resultados() {

    // preority_queue creado para ordenar los resultados de manera descendente.
    // En el pair, el int se refiere al dígito y el double a la probabilidad de que la imagen le pertenezca a ese dígito.
    priority_queue<pair<int,double>, vector<pair<int,double>>,
            function<bool(const pair<int, double>& a, const pair<int,double>& b)>> digitos (
            [](const pair<int, double>& a, const pair<int,double>& b){
                return a.second < b.second;
            }
    );

    for(int i=0; i<10; ++i){
        digitos.emplace(i,capfinal[i]);
    }

    // Asigno a un vector (normal) los datos ordenados previamente.
    while(!empty(digitos)){
        resultado.push_back(digitos.top());
        digitos.pop();
    }

    double sum{};

    // Obtengo los resultados
    for(const auto&[f,s] : resultado){
        cout<<"Digito "<<f<<": "<<s<<endl;
        sum+=s;
    }
    cout<<"\n\n";

    // Se utilizó para verificar que la suma de todas las probabilidades da 1 (100%).
    cout<<"Probabilidad total: "<<sum<<endl;

    // Se utilizo para visualizar el costo que obtuvo el dígito que se buscaba al final del proceso.
    cout<<"Costo total: "<<error<<endl;
}