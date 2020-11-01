#include "../../../stonne_works/stonne/stonne/stonne_linker_src/stonne_linker.h"
#include "../../../stonne_works/stonne/stonne/include/Config.h"

TVM_REGISTER_GLOBAL("tvm.contrib.stonne.conv2d.forward")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
        std::string path_to_arch_file = args[0];

        //Creating config  to find out if we are going to run a dense or sparse simulation
        Config stonne_cfg;
        if (path_to_arch_file != "")
        {
            stonne_cfg.loadFile(path_to_arch_file);
        }

        simulateDenseConvForward(
            layer_name,
            input_raw,
            weight_raw,
            ret[0],
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
}