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
            bool parsingSuccessful = reader.parse(f, root);
            if (!parsingSuccessful)
            {
                // report to the user the failure and their locations in the document.
                std::cout << "Failed to parse configuration\n"
                          << reader.getFormattedErrorMessages();
                return;
            }
            f.close();
            // Add in the recorded cost
            if (root["tuning_name"] == tuning_name)
            {
                root["value"].append(cost);
            }
            else
            {
                // Create new member and insert array with one value
                Json::Value content(Json::arrayValue);
                content.append(cost);
                root["value"] = content;

                // Change tuning name variable
                root["tuning_name"] = tuning_name;
            }
            // Write output
            std::ofstream fout(filename);
            writer.write(fout, root);
        }

        void transpose(float *src, float *dst, const int N, const int M) {
            // Tranpose a matrix
            #pragma omp parallel for
            for(int n = 0; n<N*M; n++) {
                int i = n/N;
                int j = n%N;
                dst[n] = src[M*j + i];
            }
        }

        //Inspired from Berkeley Vision's Caffe, modified to suit STONNE
        //https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
        // Function uses casting from int to unsigned to compare if value of
        // parameter a is greater or equal to zero and lower than value of
        // parameter b. The b parameter is of type signed and is always positive,
        // therefore its value is always lower than 0x800... where casting
        // negative value of a parameter converts it to value higher than 0x800...
        // The casting allows to use one condition instead of two.
        inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
        {
            return static_cast<unsigned>(a) < static_cast<unsigned>(b);
        }

        void im2col_cpu(const float *data_im, const int channels,
                        const int height, const int width, const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        const int dilation_h, const int dilation_w,
                        float *data_col)
        {
            const int output_h = (height + 2 * pad_h -
                                  (dilation_h * (kernel_h - 1) + 1)) /
                                     stride_h +
                                 1;
            const int output_w = (width + 2 * pad_w -
                                  (dilation_w * (kernel_w - 1) + 1)) /
                                     stride_w +
                                 1;
            const int channel_size = height * width;
            int count = 0;
            for (int channel = channels; channel--; data_im += channel_size)
            {
                for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
                {
                    for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
                    {
                        int input_row = -pad_h + kernel_row * dilation_h;
                        for (int output_rows = output_h; output_rows; output_rows--)
                        {
                            if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                            {
                                for (int output_cols = output_w; output_cols; output_cols--)
                                {
                                    *(data_col++) = 0;
                                    count++;
                                }
                            }
                            else
                            {
                                int input_col = -pad_w + kernel_col * dilation_w;
                                for (int output_col = output_w; output_col; output_col--)
                                {
                                    if (is_a_ge_zero_and_a_lt_b(input_col, width))
                                    {
                                        *(data_col++) = data_im[input_row * width + input_col];
                                            count++;
                                    }
                                    else
                                    {
                                        *(data_col++) = 0;
                                count++;
                                    }
                                    input_col += stride_w;
                                }
                            }
                            input_row += stride_h;
                        }
                    }
                }
            }
            std::cout << "Count " << count << std::endl;
        }
        

        Stonne *denseConvolution(
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
            int strides_x,
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
            Stonne *stonne_instance = simulateDenseConvForward(
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
                strides_x,
                pad_x,
                pad_y,
                path_to_tile,
                stonne_config);
            return stonne_instance;
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
            int strides_x,
            int strides_y,
            int pad_x,
            int pad_y,
            int dilation_x,
            int dilation_y,
            float sparsity_level,
            DLTensor *input,
            DLTensor *weight,
            DLTensor *output,
            Config stonne_config)
        {
            std::string layer_name = "Conv2dLayerSparse";

            // Calculate im2col output as h0 * w0 * R * S * C
            int h0 = (X + 2 * pad_x -(dilation_x * (R - 1) + 1)) /strides_x +1;
            int w0 = (Y + 2 * pad_y -(dilation_y * (S - 1) + 1)) /strides_y +1;

            // Get input and convert to im2col
            float *input_raw = static_cast<float *>(input->data);
            float im2col_array[h0*w0*R*S*C];
            float *input_im2col = im2col_array;

            // Convert weight and output files to be STONNE compatible
            float *weight_raw = static_cast<float *>(weight->data);
            float *output_raw = static_cast<float *>(output->data);
                
            // Note that since STONNE only supports sparse GEMM operations, we have to
            // turn the input to im2col format and
            // run a GEMM operation instead a CONVOLUTION    
            std::cout << "Run im2col" << std::endl;
            im2col_cpu(
                input_raw,
                C,
                X,
                Y,
                R,
                S,
                pad_x,
                pad_y,
                strides_x,
                strides_y,
                dilation_x,
                dilation_y,
                input_im2col);
            
            // Getting GEMM dimensions
            int gemm_M = K;
            int gemm_K = R*S*C;
            int gemm_N = h0*w0;

            simulateSparseGemmForward(
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
            return;
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
                int strides_x = args[11];
                int strides_y = args[12];
                int pad_x = args[13];
                int pad_y = args[14];
                int dilation_x = args[15];
                int dilation_y = args[16];
                std::string path_to_tile = args[17];
                int sparsity_ratio = args[18];
                bool tune = args[19];
                std::string tuning_name = args[20];
                std::string costs_path = args[21];
                bool stats = args[22];
                DLTensor *input = args[23];
                DLTensor *weight = args[24];
                DLTensor *output = args[25];

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

                Stonne *stonne_instance;
                if (stonne_config.sparsitySupportEnabled())
                {
                    // Convert sparsity ratio to %
                    float sparsity_ratio_float = sparsity_ratio / 100;
                    std::cout << "K init :" << K << std::endl;
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
                        strides_x,
                        strides_y,
                        pad_x,
                        pad_y,
                        dilation_x,
                        dilation_y,
                        sparsity_ratio_float,
                        input,
                        weight,
                        output,
                        stonne_config);
                }
                else if (!stonne_config.convOperationSupported()) { 
                    // If CONV itself is not supported, 
                    // run it as a GEMM (e.g., the TPU)

                    // Convert weight and output files to be STONNE compatible
                    float *weight_raw = static_cast<float *>(weight->data);
                    float *output_raw = static_cast<float *>(output->data);

                    // Calculate im2col output as h0 * w0 * R * S * C
                    int h0 = (X + 2 * pad_x -(dilation_x * (R - 1) + 1)) /strides_x +1;
                    int w0 = (Y + 2 * pad_y -(dilation_y * (S - 1) + 1)) /strides_y +1;

                    // Get input and convert to im2col
                    float *input_raw = static_cast<float *>(input->data);
                    float im2col_array[h0*w0*R*S*C];
                    float *input_im2col = im2col_array;
                    im2col_cpu(
                        input_raw,
                        C,
                        X,
                        Y,
                        R,
                        S,
                        pad_x,
                        pad_y,
                        strides_x,
                        strides_y,
                        dilation_x,
                        dilation_y,
                        input_im2col);

                    // Tranpose the result for the TPU
                    float im2col_array_tranposed[h0*w0*R*S*C];
                    float *input_im2col_t = im2col_array_tranposed;
                    transpose(input_im2col, input_im2col_t, R*S*C, h0*w0);

                    // Getting GEMM dimensions
                    int gemm_M = K;
                    int gemm_K = R*S*C;
                    int gemm_N = h0*w0;
                    simulateDenseGemmForward("TPU", input_im2col_t, weight_raw, output_raw, N, G, gemm_M, gemm_K, gemm_N, path_to_tile, stonne_config);
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
                        strides_x,
                        pad_x,
                        pad_y,
                        input,
                        weight,
                        output,
                        path_to_tile,
                        stonne_config);

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
                }
            });

    } // namespace contrib
} // namespace tvm
