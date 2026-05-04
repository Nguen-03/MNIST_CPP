#include "include/CNN.h"

int main(){
    vector<pair<int, Vector3D>> data;
    for (int i = 0; i < 500; ++i){
        data.emplace_back(0, Vector3D(1, 28, 28));
    }
    read_binary_images("dataset/test.idx", data, 500, 28, 28);
    CNN *model = new CNN();
    for (int i = 0; i < 20; ++i){ // predict 20 images
        model->predict("lmao.bin", data[i]);
        cout << endl;
    }
}