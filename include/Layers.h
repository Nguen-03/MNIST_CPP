#ifndef LAYERS_H
#define LAYERS_H
#include "utils.h"
#include <fstream>
using namespace std;

class Layers{
    public:
        virtual ~Layers() = default;
        virtual Vector3D forward(Vector3D &input) = 0;
        virtual Vector3D backward(Vector3D &grads) = 0;
        virtual void update(double lr) = 0;
        virtual void zero_grads() = 0;
        virtual void save(ofstream &file) = 0;
        virtual void load(ifstream &file) = 0;
};
#endif