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

NeuronalNetwork::~NeuronalNetwork() = default;

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

void NeuronalNetwork::inicializacion_Xavier() { // Para conseguir valores apropiados para mis matrices de pesos
    factor_xavier = sqrt(2.0/(static_cast<int>(cap1.size()) + n3));
}

void NeuronalNetwork::inicializar_capas(const Mat& img) {

    VectorXd guardar1 = MatToVector(img);
    VectorXdToUnorderedMap(guardar1);
    int divisor = minima_frecuencia_px();
    inicializacion_Xavier();

    //Cuando son pocas capas, puede usarse solo random. Pero decidi usar el factor xavier que se utiliza cuando son
    //muchas capas (estoy usando solo 2 capas ocultas).

    cap1 = neuronas_entrada(divisor);
    w1 = MatrixXd::Random(n1, cap1.size()) ;
    z1 = VectorXd::Zero(n1); //el factor xavier es para definir los pesos iniciales.

    //ver el problema del tamaño de los out1 y de los sesgos.

    b1 = VectorXd::Zero(n1);
    w2 = MatrixXd::Random(n2, n1);
    cap2 = VectorXd::Zero(n1); //en esta seccion defino un tamaño fijo para cada matriz o vector Eigen.
    z2 = VectorXd::Zero(n1);

    b2 = VectorXd::Zero(n2);
    w3 = MatrixXd::Random(n3, n2) ;
    cap3 = VectorXd::Zero(n2);
    z3 = VectorXd::Zero(n2);

    b3 = VectorXd::Zero(n3);
    capfinal = VectorXd::Zero(n3);

    //one - hot
    etiqueta_real = VectorXd::Zero(n3);
    etiqueta_real[indice] = 1;
}

NeuronalNetwork::NeuronalNetwork(const Mat& img, int n1_1, int n2_1, int n3_1, int iter, int ind){
    n1 = n1_1;
    n2 = n2_1;
    n3 = n3_1;
    iteraciones = iter;
    indice = static_cast<Index>(ind);

    inicializar_capas(img);
}

// Funciones de activación

double NeuronalNetwork::sigmoid(double x) { //funcion de activacion (binaria - 2 capas solamente)
    return 1.0 / (1.0 + exp(-x));

    //se utiliza para conseguir el vector intermedio entre la ambas capas.
}

VectorXd  NeuronalNetwork::softMax(const VectorXd& x) { // funcion de activacion para la cada de salida
    VectorXd exp_xd = x.array().exp(); //para convertir los valores en una distribucion de probabilidad (multiclase - 3 o más capas)
    double sum_xd = exp_xd.sum();

    return exp_xd / sum_xd;
    //se utiliza para conseguir el vector intermedio entre la penultima capa y la capa final.
}

VectorXd NeuronalNetwork::Relu(const VectorXd& eigen) {
    return eigen.cwiseMax(0.0);
}

// Derivadas de las funciones de activación

double NeuronalNetwork::derivada_sigmoid(double x) {
    /* Hay 2 formas: Derivando simplemente el sigmoide
    return (exp(-x)/pow((1+exp(-x)),2));
    */

    // O... (más optima por no usar el pow())
    return sigmoid(x) * (1.0 - sigmoid(x));
}

VectorXd NeuronalNetwork::derivada_softMax(const Eigen::VectorXd &x) {
    return softMax(x).array() * (1 - softMax(x).array());

    /*  Esto es equivalente:
            VectorXd SM = softMax(x);
        VectorXd retorno= VectorXd::Zero(SM.size());

        for(Index i=0; i<SM.size();++i ){
           retorno[i] = SM[i] * (1 - SM[i]);
        }

        return retorno;
     */
}

void NeuronalNetwork::forwardPropagation() {

    // z = wx + b

    //bi = vector de sesgo de capi+i

    z1 = w1 * cap1 + b1;
    cap2 = (z1).unaryExpr([](double x) { return sigmoid(x); }) ;

    z2 = w2 * cap2 + b2;
    cap3 = (z2).unaryExpr([](double x) { return sigmoid(x); }) ;

    z3 = w3 * cap3 + b3;

    capfinal = softMax(z3) ;

    // para conseguir la capai = función de activación (capa_intermedia_i-1 + matriz_de_sesgos_i)
}

Index NeuronalNetwork::Prediccion() {
    Index maxIndex;
    capfinal.maxCoeff(&maxIndex); //saca el índice con mayor probabilidad.

    return maxIndex;
}

double NeuronalNetwork::Funcion_Perdida(const VectorXd& Funcion_activacion) const {
    //tambien llamada el Coste, función de perdida o ...

    //esta función de perdida = L = - sumatoria en j=1 hasta el #clases de (yj ln(^yj))
    //esto se reduce solamente a = L = -ln(^yk) = -ln(probabilidad de la clase correcta despues de aplicar la FA softmax)

    VectorXd FA = softMax(Funcion_activacion);
    double prob = FA[indice];
    return -log(prob + 1e-10); //eñ 1e-10 es para evitar el log(0).
}

VectorXd NeuronalNetwork::Gradiante_Funcion_Perdida(const VectorXd& z) {
    //Para calcular el Gradiante del error de perdida (Cross Entropy) utilizando la FA de softmax:
    // Osea gradiante del costo / gradiante de z = Vector de la predicción del modelo - etiqueta real

    //Tengo que asegurarme que ambos vectores tienen las mismas dimensiones.

    VectorXd softMaxs = softMax(z);
    return softMaxs - etiqueta_real;
}

void NeuronalNetwork::backPropagation() {
    forwardPropagation();

    //En z = wx + b (fórmula aplicada en la propagación para adelante)
    // Grad_FP3 * cap3.transpose() = aL/aw = aL/az * az/aw || az/aw = transpuesta de la capa
    // Grad_FP3 * 1 = aL/ab = aL/az * az/ab || az/ab = 1

    VectorXd Grad_FP3 = Gradiante_Funcion_Perdida(z3);
    w3 = w3 - tasa_aprendizaje * (Grad_FP3 * cap3.transpose());
    b3 = b3 - tasa_aprendizaje * (Grad_FP3 * 1);

    VectorXd Grad_FP2 = (w3.transpose() * Grad_FP3).cwiseProduct(z2.unaryExpr([](double x) { return derivada_sigmoid(x);}));
    w2 = w2 - tasa_aprendizaje * (Grad_FP2 * cap2.transpose());
    b2 = b2 - tasa_aprendizaje * (Grad_FP2 * 1);

    VectorXd Grad_FP1 = (w2.transpose() * Grad_FP2).cwiseProduct(z1.unaryExpr([](double x){ return derivada_sigmoid(x);}));
    w1 = w1 - tasa_aprendizaje * (Grad_FP1 * cap1.transpose());
    b1 = b1 - tasa_aprendizaje * (Grad_FP1 * 1);

    error = Funcion_Perdida(capfinal); // calculo del error.
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
