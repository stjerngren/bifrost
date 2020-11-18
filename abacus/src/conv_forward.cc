#include "../../../stonne_works/stonne/stonne/stonne_linker_src/stonne_linker.h"
#include "../../../stonne_works/stonne/stonne/include/Config.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.conv2d.forward")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int R = args[1];
                int S = args[2];
                int C = args[3];
                int K = args[4];
                int G = args[5];
                int N = args[6];
                int X = args[7];
                int Y = args[8];
                int H_out = args[9];
                int W_out = args[10];
                int strides = args[11];
                int pad_x = args[12];
                int pad_y = args[13];
                std::string path_to_tile = args[14];
                DLTensor *input = args[15];
                DLTensor *weight = args[16];
                DLTensor *output = args[17];

                //Creating config  to find out if we are going to
                // run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }

                // Run different types of convolutions depending
                // on whether sparsity is suported
                if (stonne_config.sparsitySupportEnabled())
                {
                    // Run a sparse forward convolution
                    sparseConvolution(
                        R,
                        S,
                        C,
                        K,
                        G,
                        N,
                        X,
                        Y,
                        H_out,
                        W_out,
                        input,
                        weight,
                        output,
                        stonne_config )
                }
                else
                {
                    // Run a dense forward convolution
                    denseConvolution(
                        R,
                        S,
                        C,
                        K,
                        G,
                        N,
                        X,
                        Y,
                        H_out,
                        W_out,
                        strides,
                        pad_x,
                        pad_y,
                        input,
                        weight,
                        output,
                        path_to_tile,
                        stonne_config)
                }
            });

    } // namespace contrib
} // namespace tvm

void denseConvolution(
    int R,
    int S,
    int C,
    int K,
    int G,
    int N,
    int X,
    int Y,
    int H_out,
    int W_out,
    int strides,
    int pad_x,
    int pad_y,
    DLTensor *input,
    DLTensor *weight,
    DLTensor *output,
    std::string path_to_tile,
    Config stonne_config)
{
    // Cast pointers so they can be fed into stonne
    float *input_raw = static_cast<float *>(input->data);
    float *weight_raw = static_cast<float *>(weight->data);
    float *output_raw = static_cast<float *>(output->data);

    // Choose name for output statistics
    std::string layer_name = "Conv2dLayerDense";

    // Run the simulated forward convolution
    simulateDenseConvForward(
        layer_name,
        input_raw,
        weight_raw,
        output_raw,
        R,
        S,
        C,
        K,
        G,
        N,
        X,
        Y,
        H_out,
        W_out,
        strides,
        pad_x,
        pad_y,
        path_to_tile,
        stonne_config);
}

void sparseConvolution(
    int R,
    int S,
    int C,
    int K,
    int G,
    int N,
    int X,
    int Y,
    int H_out,
    int W_out,
    DLTensor *input,
    DLTensor *weight,
    DLTensor *output,
    Config stonne_config, )
{
    float sparsity_level = 0;
    std::string layer_name = "Conv2dLayerSparse";

    std::cout << "Sparsity support enabled with ratio" << sparsity_level << std::endl;

    // All the channels. Note this could not be the
    // same in weight.sizes[1] (i.e., filter channels)
    // as the groups could reduce these last ones.
    // In this case, we send the complete number of input channels, and the
    // callee will have to be aware of this and run C/G if  groups exist.

    torch::Tensor input_im2col = F::unfold(
        input,
        F::UnfoldFuncOptions({R, S}).padding(padding).stride(stride).dilation(
            dilation)); // This function returns a 3D tensor [N, R*S*C,
                        // number_of_outputs]
    // Getting raw data
    float *KN_input_raw = (float *)input_im2col.data_ptr();

    float *MK_weight_raw = (float *)weight.data_ptr();
    // Creating output tensor

    torch::Tensor output = torch::rand({N, K, H_out, W_out});
    float *output_raw = (float *)output.data_ptr();
    // Note that since STONNE only supports sparse GEMM operation, we have to
    // turn
    // the input to im2col format and run a GEMM operation instead a CONVOLUTION
    // Getting GEMM dimensions
    // MK matrix are the weights
    int gemm_M = K;                       // Number of filters (weight.sizes()[0];) (i.e., rows MK)
    int gemm_K = input_im2col.sizes()[1]; // window size (i.e., columns MK)
    int gemm_N = input_im2col.sizes()[2]; // 0 is batch dim, 1 is K
    simulateSparseGemmForward(
        layer_name,
        KN_input_raw,
        MK_weight_raw,
        output_raw,
        N,
        G,
        gemm_M,
        gemm_K,
        gemm_N,
        sparsity_level,
        stonne_config,
        MK_STA_KN_STR); // Keeping MK stationary as they are the weights
                        // Cast the input and output data into float pointer arrays
                        // which are compatible with stonne
}