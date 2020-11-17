#include "../../../stonne_works/stonne/stonne/stonne_linker_src/stonne_linker.h"
#include "../../../stonne_works/stonne/stonne/include/Config.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.test")
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

                std::cout << input->strides;
                std::cout << (input->shape)[0];
                std::cout << (input->shape)[1];
                std::cout << (input->shape)[2];
                std::cout << (input->shape)[3];
                std::cout << "\n offset \n";
                std::cout << input->byte_offset;
                std::cout << "\n";

                float *data = static_cast<float *>(input->data);
                int ndim = input->ndim;

                for (int i = 0; i < 37; i++)
                    std::cout << *(data + i) << ", ";

                std::cout << "\n converted \n";
                output->ndim = input->ndim;
                output->shape = input->shape;
                output->data = data;
                output->byte_offset = input->byte_offset;
                for ( int i = 0; i < 34; i++)
                    std::cout << *(static_cast<float*>(output->data) + i) << ", ";
                std::cout << (output->shape)[1];
                std::cout << (output->shape)[2];
                std::cout << (output->shape)[3];
                std::cout << (output->shape)[0];

  
            });

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

                float *input_raw = static_cast<float *>(input->data);
                float *weight_raw = static_cast<float *>(weight->data);

                float output_array[32];
                for (int i = 0; i < 32; i++)
                {
                    output_array[i] = 0.0;
                }
                float *output_raw = output_array;

                std::string layer_name = "Conv2dLayer";
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

                output->shape = input->shape;
                output->byte_offset = input->byte_offset;
                output->data = output_raw;
                for (int i = 0; i < 34; i++)
                    std::cout << *(static_cast<float *>(output->data) + i) << ", ";
            });

    } // namespace contrib
} // namespace tvm