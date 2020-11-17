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
                // TODO: Add sparse support

                // Cast the input and output data into float pointer arrays
                // which are compatible with stonne
                float *input_raw  = static_cast<float *>(input->data);
                float *weight_raw = static_cast<float *>(weight->data);
                float *output_raw = static_cast<float *>(output->data);

                // Choose name for output statistics 
                std::string layer_name = "Conv2dLayer";

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
            });

    } // namespace contrib
} // namespace tvm