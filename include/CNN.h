#ifndef CNN_H
#define CNN_H

#include <iostream>
#include "Layers.h"
#include "Conv2D.h"
#include "ReLU.h"
#include "Linear.h"
#include "MaxPool.h"
#include "Flatten.h"
#include "CrossEntropyLoss.h"
#include <memory>
#include <iomanip>
#include <cstdlib>
using namespace std;

class CNN{
    public:
        vector<Layers*> layers;
        CrossEntropyLoss criterion; 
        CNN();
        ~CNN();
        Vector3D forward(Vector3D &input);
        Vector3D backward(Vector3D &grads);
        void update(double lr); //update learning rate
        double evaluate(vector<pair<int, Vector3D>> &data); //evaluate model while training
        void train(vector<pair<int, Vector3D>> &train, vector<pair<int, Vector3D>> &val, int batch_size, int epochs, string file_path);
        void zero_grads();  //reset gradients
        void save(string file_path); //save weights, biases
        void load(string file_path); //load weights, biases for prediction
        void predict(string path, pair<int, Vector3D> &data);
};

#endif