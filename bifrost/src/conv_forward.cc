// STONNE
#include "../../../stonne_works/stonne/stonne/stonne_linker_src/stonne_linker.h"
#include "../../../stonne_works/stonne/stonne/include/Config.h"
#include "../../../stonne_works/stonne/stonne/include/STONNEModel.h"

// TVM
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

// JSONCPP
#include "json/json.h"
#include "json/json-forwards.h"

#include <fstream>
#include <string>
#include <iostream>



// Stonne variable taxonomy
// -R: Number of flter rows
// -S: Number of filter columns
// -C: Number of filter and input channels
// -K: Number of filters and output channels
// -G: Number of groups
// -N: Number of inputs (Only 1 is supported so far)
// -X: Number of input rows
// -Y: Number of input columns
// -X_: Number of output columns
// -Y_: Number of output columns

namespace tvm
{
    namespace contrib
    {

        using namespace runtime;
        
        void reportCost(
            std::string tuning_name, 
            std::string filename, 
            int cost)
        {
            // Intialise the JSONCPP variables
            Json::Value root;   
            Json::Reader reader;
            Json::StyledStreamWriter writer;

            // Read the file
            std::ifstream f(filename);

            // Parse the file
            bool parsingSuccessful = reader.parse( f, root );
            if ( !parsingSuccessful )
            {
                // report to the user the failure and their locations in the document.
                std::cout  << "Failed to parse configuration\n"
                        << reader.getFormattedErrorMessages();
                return;
            }
            f.close();
            // Add in the recorded cost
            if (root["tuning_name"] == tuning_name) {
                root["value"].append(cost);
            } else {
                // Create new member and insert array with one value
                Json::Value content(Json::arrayValue);
                content.append(cost);
                root["value"] = content;

                // Change tuning name variable
                root["tuning_name"] = tuning_name;

            }
            // Write output
            std::ofstream fout(filename);
            writer.write(fout,root);

        }

        float im2col_get_pixel(

            float *im,
            int height,
            int width,
            int channels,
            int row,
            int col,
            int channel,
            int pad)
        {
            row -= pad;
            col -= pad;

            if (row < 0 || col < 0 ||
                row >= height || col >= width)
                return 0;
            return im[col + width * (row + height * channel)];
        }

        //Inspired from Berkeley Vision's Caffe, modified to suit STONNE
        //https://github.com/BVLC/caffe/blob/master/LICENSE
        void im2col_cpu(
            float *data_im,
            int channels,
            int height,
            int width,
            int ksize,
            int stride,
            int pad,
            float *data_col)
        {
            int c, h, w;
            int height_col = (height + 2 * pad - ksize) / stride + 1;
            int width_col = (width + 2 * pad - ksize) / stride + 1;

            int channels_col = channels * ksize * ksize;

            for (c = 0; c < channels_col; ++c)
            {
                int w_offset = c % ksize;
                int h_offset = (c / ksize) % ksize;
                int c_im = c / ksize / ksize;
                for (h = 0; h < height_col; ++h)
                {
                    for (w = 0; w < width_col; ++w)
                    {
                        int im_row = h_offset + h * stride;
                        int im_col = w_offset + w * stride;
                        int col_index = (c * height_col + h) * width_col + w;
                        data_col[col_index] = im2col_get_pixel(
                            data_im,
                            height,
                            width,
                            channels,
                            im_row,
                            im_col,
                            c_im,
                            pad);
                    }
                }
            }
        }

        Stonne* denseConvolution(
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
            Config stonne_config
            )
        {
            // Cast pointers so they can be fed into stonne
            float *input_raw = static_cast<float *>(input->data);
            float *weight_raw = static_cast<float *>(weight->data);
            float *output_raw = static_cast<float *>(output->data);

            // Choose name for output statistics
            std::string layer_name = "Conv2dLayerDense";

            // Run the simulated forward convolution
            Stonne* stonne_instance = simulateDenseConvForward(
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
                stonne_config
                );
            return stonne_instance;
        }

        Stonne* sparseConvolution(
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
            int pad,
            DLTensor *input,
            DLTensor *weight,
            DLTensor *output,
            Config stonne_config)
        {
            float sparsity_level = 0;
            std::string layer_name = "Conv2dLayerSparse";

            std::cout << "Sparsity support enabled with ratio" << sparsity_level << std::endl;

            // All the channels. Note this could not be the
            // same in weight.sizes[1] (i.e., filter channels)
            // as the groups could reduce these last ones.
            // In this case, we send the complete number of input channels, and the
            // callee will have to be aware of this and run C/G if  groups exist.

            //torch::Tensor input_im2col = F::unfold(
            //    input,
            //    F::UnfoldFuncOptions({R, S}).padding(padding).stride(stride).dilation(
            //        dilation)); // This function returns a 3D tensor [N, R*S*C,
            // number_of_outputs]
            // Getting raw data
            //float *KN_input_raw = (float *)input_im2col.data_ptr();

            float *input_raw = static_cast<float *>(input->data);
            float im2col_array[C * K * K];
            float *input_im2col = im2col_array;
            float *weight_raw = static_cast<float *>(weight->data);
            float *output_raw = static_cast<float *>(output->data);

            // Note that since STONNE only supports sparse GEMM operations, we have to
            // turn the input to im2col format and
            // run a GEMM operation instead a CONVOLUTION
            im2col_cpu(
                input_raw,
                C,
                X,
                Y,
                K,
                strides,
                pad,
                input_im2col);

            // Getting GEMM dimensions
            // MK matrix are the weight
            std::cout << "test sparse gemm" << std::endl;

            int gemm_M = K;
            int gemm_K = C * K * K;
            int gemm_N = 0;

            std::cout << "perform sparse gemm" << std::endl;

            Stonne* stonne_instance = simulateSparseGemmForward(
                layer_name,
                input_im2col,
                weight_raw,
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
            return stonne_instance;
        }

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
                int sparsity_ratio = args[15];
                bool tune = args[16];
                std::string tuning_name = args[17];
                std::string costs_path = args[18];
                bool stats = args[19];
                DLTensor *input = args[20];
                DLTensor *weight = args[21];
                DLTensor *output = args[22];

                //Creating config  to find out if we are going to
                // run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }

                // TODO: Make stats printing optional by choosing a variable
                // Turn of stats printing
                stonne_config.print_stats_enabled = stats;

                // Run different types of convolutions depending
                // on whether sparsity is suported

                Stonne* stonne_instance; 
                if (stonne_config.sparsitySupportEnabled())
                {   
                    // Convert sparsity ratio to %
                    float sparsity_ratio_flaot = sparsity_ratio/100;
                    
                    // Run a sparse forward convolution
                    stonne_instance = sparseConvolution(
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
                        input,
                        weight,
                        output,
                        stonne_config);           
                }
                else
                {
                    // Run a dense forward convolution
                    stonne_instance = denseConvolution(
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
                        stonne_config);
                }

                // If the hardware is being tuned, report the cost
                if (tune) 
                {
                    reportCost(
                        tuning_name,
                        costs_path,
                        stonne_instance->n_cycles

                    );
                }
                delete stonne_instance;

                
            });

    } // namespace contrib
} // namespace tvm
