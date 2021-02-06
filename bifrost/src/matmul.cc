// STONNE
#include "stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// JSONCPP
#include "json/json.h"
#include "json/json-forwards.h"

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;

        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.matmul")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int M = args[1]; // Batch size
                int K = args[2]; // Number of input neurons
                int N = args[3]; // Number of output neurons
                std::string path_to_tile = args[4];
                int sparsity_ratio = args[5];
                bool stats = args[6];
                DLTensor *input = args[7];
                DLTensor *weight = args[8];
                DLTensor *output = args[9];
            });

    } // namespace contrib
} // namespace tvm
