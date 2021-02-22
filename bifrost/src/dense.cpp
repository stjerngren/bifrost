// STONNE
#include "stonne_linker.h"
#include "Config.h"
#include "STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// Cost function
#include "include/cost.h"

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        TVM_REGISTER_GLOBAL("tvm.contrib.stonne.dense")
            .set_body([](TVMArgs args, TVMRetValue *ret) {
                std::string path_to_arch_file = args[0];
                int M = args[1]; // Batch size
                int K = args[2]; // Number of input neurons
                int N = args[3]; // Number of output neurons
                std::string path_to_tile = args[4];
                int sparsity_ratio = args[5];
                bool stats = args[6];
                bool tune = args[7];
                bool tune_psums = args[8];
                std::string tuning_name = args[9];
                std::string costs_path = args[10];
                DLTensor *input = args[11];
                DLTensor *weight = args[12];
                DLTensor *output = args[13];

                // Add some way to specify layer names
                std::string layer_name = "Test";

                // Init tuning cycle variable
                int cycles;

                //Here starts the function
                //Creating config file to find out if we are going to run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }
                stonne_config.print_stats_enabled = stats;

                // Cast pointers so they can be fed into stonne
                float *input_raw = static_cast<float *>(input->data);
                float *weight_raw = static_cast<float *>(weight->data);
                float *output_raw = static_cast<float *>(output->data);
                if (tune_psums)
                {

                    float *KN_matrix_raw = weight_raw;
                    float *MK_matrix_raw = input_raw;
                    int N = 1;
                    int G = 1;
                    int gemm_M = M;
                    int gemm_K = K;
                    int gemm_N = N;

                    if (path_to_tile == "")
                    {
                        std::cout << "Tile file parameters must be specified" << std::endl;
                        exit(1);
                    }

                    // Count the number of osums
                    int cost = 0;
                    //Loading the tile
                    Tile tile(path_to_tile);

                    int gemm_M_grouped = gemm_M / G;
                    int gemm_K_grouped = gemm_K / G;
                    int gemm_N_grouped = gemm_N;

                    int weight_offset_ = gemm_M_grouped * gemm_K_grouped;
                    int col_offset_ = gemm_K_grouped;
                    int out_offset_ = gemm_M_grouped * gemm_N_grouped;
                    float *sub_KN_dense_matrix = new float[gemm_N_grouped * gemm_K_grouped];

                    //Setting accelerator parameters. Mapping GEMM into a conv tile and layer
                    //int S = gemm_K_grouped;
                    //int K = gemm_N_grouped;
                    //int N = gemm_M_grouped;

                    //Tile parameters
                    int T_K = tile.get_T_S();
                    int T_N = tile.get_T_K();
                    int T_M = tile.get_T_N();

                    for (int n = 0; n < N; n++)
                    {
                        std::string layer_name_batch = layer_name + "_batch_" + std::to_string(n);
                        for (int g = 0; g < G; g++)
                        { //Every group is calculated with a different GEMM
                            std::string layer_name_group = (layer_name_batch + ("_group_" + std::to_string(g)));
                            float *pointer_MK_dense_matrix = (float *)MK_matrix_raw + weight_offset_ * g;
                            float *pointer_KN_dense_matrix = (float *)KN_matrix_raw + col_offset_ * g;
                            //Since the offset in this case is per column, we have to use a submatrix.
                            for (int i = 0; i < gemm_N_grouped; i++)
                            {
                                for (int j = 0; j < gemm_K_grouped; j++)
                                {
                                    float *sub_pointer = (float *)KN_matrix_raw + col_offset_ * g;
                                    sub_KN_dense_matrix[i * gemm_K_grouped + j] = sub_pointer[i * gemm_K + j];
                                }
                            }

                            pointer_KN_dense_matrix = sub_KN_dense_matrix;
                            float *acc_output = (float *)output_raw + out_offset_ * g;

                            Stonne *stonne_instance = new Stonne(stonne_config); //Creating the instance of the simulator
                            stonne_instance->loadFCLayer(layer_name_group, gemm_M_grouped, gemm_K_grouped, gemm_N_grouped, (address_t)pointer_MK_dense_matrix, (address_t)pointer_KN_dense_matrix, (address_t)acc_output);
                            stonne_instance->loadFCTile(T_K, T_M, T_N);
                            cost = cost + stonne_instance->mem->getPsums();    
                            delete stonne_instance;
                        }
                    }
                    delete[] sub_KN_dense_matrix;

                    reportCost(
                        tuning_name,
                        costs_path,
                        cost

                    );
                    return;
                }

                if (stonne_config.sparsitySupportEnabled())
                {
                    // Convert sparsity ratio to %
                    float sparsity_ratio_float = sparsity_ratio / 100;
                    cycles = simulateSparseGemmForward(
                        layer_name,
                        weight_raw,
                        input_raw,
                        output_raw,
                        1, 1, M, K, N,
                        sparsity_ratio_float,
                        stonne_config,
                        MK_STR_KN_STA);
                }
                else
                {
                    cycles = simulateDenseGemmForward(
                        layer_name,
                        weight_raw,
                        input_raw,
                        output_raw,
                        1, 1, M, K, N,
                        path_to_tile,
                        stonne_config);
                }
                if (tune)
                // If the hardware is being tuned, report the cost
                {
                    reportCost(
                        tuning_name,
                        costs_path,
                        cycles

                    );
                } else {
                    reportTotalCycles(
                        tuning_name,
                        "bifrost_temp/cycles.json",
                        cycles
                    );
                }
            });

    } // namespace contrib
} // namespace tvm
