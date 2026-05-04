#ifndef CROSS_ENTROPY_LOSS
#define CROSS_ENTROPY_LOSS
#include "utils.h"
#include <algorithm>
#include <cmath>
#include "Layers.h"
class CrossEntropyLoss{
    public:
        int label_cache;
        vector<double> p;
        Vector3D grads;
        CrossEntropyLoss();
        ~CrossEntropyLoss();
        double loss(const vector<double> &logits, int label);
        Vector3D backward();
        void zero_grads();
};

#endif