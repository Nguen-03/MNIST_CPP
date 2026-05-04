#include "CrossEntropyLoss.h"
#include <iostream>

CrossEntropyLoss::CrossEntropyLoss(){}
CrossEntropyLoss::~CrossEntropyLoss(){}

double CrossEntropyLoss::loss(const vector<double> &logits ,int label){
    int n = logits.size();
    this->p = vector<double>(n);

    if (this->grads.data.empty()){
        this->grads = Vector3D(1, 1, n);
    }else
        fill(this->grads.data.begin(), this->grads.data.end(), 0.0);

    double maxVal = *max_element(logits.begin(), logits.end());
    double sum = 0.0;

    for (int i = 0; i < n; ++i){
        p[i] = exp(logits[i] - maxVal);
        sum += p[i];
         if (isnan(logits[i]) || isinf(logits[i])){
            cout << "LOGIT NAN DETECTED\n";
            exit(0);
        }
    }

    for (int i = 0; i < n; ++i)
        p[i] /= sum;

    for (int i = 0; i < n; ++i)
        grads.at(0, 0, i) = p[i];

    this->grads.at(0, 0, label) -= 1.0;
    this->label_cache = label;
    // cout << grads.data;
    return -log(p[label] + 1e-12);

}//end loss function

Vector3D CrossEntropyLoss::backward(){
    return this->grads;
}

void CrossEntropyLoss::zero_grads(){
    if (!this->grads.data.empty())
        fill(this->grads.data.begin(), this->grads.data.end(), 0.0);
}