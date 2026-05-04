#ifndef FLATTEN_H
#define FLATTEN_H
#include <iostream>
#include "utils.h"
#include "Layers.h"
using namespace std;

class Flatten : public Layers{
    public:
        int rows, cols, channels;
        Flatten() {};
        Vector3D forward(Vector3D &input) override;
        Vector3D backward(Vector3D &grad) override;
        void update(double lr) override;
        void zero_grads() override;
        void save(ofstream &file) override;
        void load(ifstream &file) override;
};
#endif