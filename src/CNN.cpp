#include "CNN.h"

CNN::CNN(){
    this->layers = {
        new Conv2D(1, 16, 3),
        new ReLU(),
        new MaxPool(2, 2),
        new Conv2D(16, 32, 3),
        new ReLU(),
        new MaxPool(2, 2),
        new Flatten(),
        new Linear(32 * 5 * 5, 128, false),
        new ReLU(),
        new Linear(128, 10, false)
    };

}

CNN::~CNN() {
    for (auto layer : layers) {
        delete layer;
    }
    layers.clear();
}

Vector3D CNN::forward(Vector3D &input){
    Vector3D x = input;

    for (auto &layer : this->layers){
        x = layer->forward(x);
        
    }
    return x;
}

Vector3D CNN::backward(Vector3D &grads){
    Vector3D x = grads;
    for (int i = this->layers.size() - 1; i >= 0; --i)
        x = layers[i]->backward(x);
    return x;
}

void CNN::update(double lr){
    for (auto &layer : this->layers)
        layer->update(lr);
}

void CNN::zero_grads(){
    for (auto &layer : this->layers)
        layer->zero_grads();
}

void CNN::save(string file_path){
    ofstream file(file_path, ios::binary);
    for (auto &layer : this->layers)
        layer->save(file);
}

void CNN::load(string file_path){

    ifstream file(file_path, ios::binary);
    
    if (file.is_open()){
        for (auto &layer : this->layers){
            layer->load(file);
        }
    }
    file.close();
}

double CNN::evaluate(vector<pair<int, Vector3D>> &data){

    int data_size = data.size();
    int correct = 0;
        // #pragma omp parallel for
    for (int i = 0; i < data_size; ++i){
        Vector3D out = this->forward(data[i].second);
        int predict = 0;
        double maxValue = out.at(0, 0, 0);
        
        for (size_t j = 1; j < (size_t)out.thirdD; ++j){

            if (out.at(0, 0, j) > maxValue){
                maxValue = out.at(0, 0, j);
                predict = j;
            }// end if

        } //end j

        if (predict == data[i].first)
            ++correct;
    } //end i
    return (double)correct / data_size;
}

void CNN::train(vector<pair<int, Vector3D>> &train, vector<pair<int, Vector3D>> &val, int batch_size, int epochs, string file_path){
    ios_base::sync_with_stdio(false);
    cout.tie(NULL);
    int train_size = train.size();
    double lr = 0.005;
    double best_accuracy = 0.0;
    

    for (int e = 0; e < epochs; ++e){
        double epoch_loss = 0.0;

        for (int i = 0; i < train_size; i += batch_size){ //loop all batches
            int riel_batch_size = min(batch_size, train_size - i);
            this->zero_grads(); // reset gradient
            
            for (int j = i; j < i + riel_batch_size; ++j){ // run 1 batch
                
                Vector3D out = forward(train[j].second);
                
                // cout << out.data << endl;
                epoch_loss += this->criterion.loss(out.data, train[j].first);
                Vector3D grads = this->criterion.backward();
                this->backward(grads);

                cout << "\rEPOCH: " << e + 1 << "/" << epochs << ": " << "Batch: " << i / batch_size << "/" << ceil(train_size / batch_size) << flush;
                cout << "\033[?25l";
                
            }//end 1 batch

            this->update(lr / riel_batch_size);

        }//end all batches (1 epoch)

        if ((e + 1) % 5 == 0)
            lr /= 2;
        
        double accuracy = this->evaluate(val);

        if (accuracy > best_accuracy){
            best_accuracy = accuracy;
            this->save(file_path); //save gradients that have the best accuracy
        }//end if
        
        cout  << ": Accuracy: " << accuracy * 100 << "%     |    " << "Loss: " << epoch_loss / train_size << endl;
        shuffle_data(train); //suffle
    } //end epochs
    
}

void CNN::predict(string path, pair<int, Vector3D> &data){
    print_image(data.second);
    this->load(path);

    Vector3D out = this->forward(data.second); //forward data to predict
    double val = out.at(0,0,0);
    int result = 0;

    for (size_t i = 1; i < (size_t)out.thirdD; ++i){
        if (out.at(0, 0, i) > val){
            val = out.at(0, 0, i);
            result = i;
        }
    }

    cout << "  ";
    for (int i = 0; i < 10; ++i)
        cout << i << "        ";
    cout << endl;

    this->criterion.loss(out.data, result);
    vector<double> conf = this->criterion.p;  //confidence

    for (auto &x : conf){
        x *= 100;
        cout << fixed <<  setprecision(2) <<  x  << "%    ";
    }
    cout << endl;
    
    for (int i = 0; i < 50; ++i)
        cout << "=";
    
    cout << endl;
    cout << "CORRECT: " << data.first << endl;
    cout << "PREDICT: " << result;
}