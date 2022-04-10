#include<torch/extension.h>

//void calculate(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
void calculate_dataingpu(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
void calculate_dataincpu(float*, float*, float*, int64_t, int64_t, int64_t, int64_t);

torch::Tensor mymultiply(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){//a:input b:kernel
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    auto ret = torch::zeros({m, n});
    int64_t size = ret.numel();
    if(input.device().type()==torch::kCPU){
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    else if(input.device().type()==torch::kCUDA){
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    //calculate(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    //calculate(a, b, ret, size, m, n);
    return ret;
}
torch::Tensor mymultiply_nobias(torch::Tensor input, torch::Tensor weight){//a:input b:kernel
    int64_t m = input.size(0);
    int64_t k = input.size(1);
    int64_t n = weight.size(1);
    auto ret = torch::zeros({m, n});
    int64_t size = ret.numel();
    if(input.device().type()==torch::kCPU){
        calculate_dataincpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    else if(input.device().type()==torch::kCUDA){
        ret = ret.cuda();
        calculate_dataingpu(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    }
    //calculate(input.data_ptr<float>(), weight.data_ptr<float>(), ret.data_ptr<float>(), size, m, n, k);
    //calculate(a, b, ret, size, m, n);
    return ret;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mymultiply", &mymultiply, "a test");
    m.def("mymultiply_nobias",&mymultiply_nobias,"gemm with no bias");
}
