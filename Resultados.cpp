#include "NeuronalNetwork.h"

VectorXd NeuronalNetwork::Sacar_Dato(const int& fila_csv) {

    ifstream csv_dato(csv_Dato);
    if (!csv_dato.is_open()) {
        cerr << "Error: No se pudo abrir el archivo CSV." << std::endl;
    }


    vector<int> px_digito;

    string line;
    bool solo_uno = false;
    int i{};

    while (getline(csv_dato, line)) {
        stringstream lineStream(line);
        string cell;

        if(i<fila_csv-1){
            ++i;
            continue;
        }
        else{
            while (getline(lineStream, cell, ',')) {
                if (!solo_uno) {
                    solo_uno = true;
                    continue;
                }
                int datos = std::stoi(cell);
                px_digito.push_back(datos);
            }
            break;
        }
    }

    VectorXd retorno = VectorXd::Zero(static_cast<Index>(px_digito.size()));

    for(int j=0; j<px_digito.size(); ++j){
        retorno[j] = px_digito[j];
    }

    return retorno;
}

void NeuronalNetwork::Prediccion(const int& fila) {

    cap1 = Sacar_Dato(fila);

    forwardPropagation(mtx);
}

// Función para obtener los resultados (Se utiliza después de la función de "Iteraciones", para obtener los
// resultados finales después el proceso)
void NeuronalNetwork::resultados(const int& fila_csv) {

    Prediccion(fila_csv);
    vector<pair<int,double>> resultado(10);

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
    int j{};
    while(!empty(digitos)){
        resultado[j] = (digitos.top());
        digitos.pop();
        ++j;
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

    sum=0;
}