#include "../../stonne_works/stonne/stonne/stonne_linker_src/stonne_linker.h"
#include <tvm/runtime/registry.h>
#include "../../stonne_works/stonne/stonne/include/Config.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <iostream>
#include <unistd.h>
#include "STONNEModel.h"
#include "Tile.h"
#include "Config.h"
#include "types.h"
#include "testbench.h"
#include "utility.h"
#include <string>

float* Transform_Ifmap_Memory (const float* bottom_data, const int C, const int X, const int Y, const int pad_x, const int pad_y);

float* Transform_Filters_Memory (const float* weights, const int K, const int G, const int C, const int R, const int S);

void Transform_Ofmap_Memory (const float* ofmap_data, float* top_data, const int K, const int X_, const int Y_);

float* Transform_Ifmap_Memory (const float* bottom_data, const int C, const int X, const int Y, const int pad_x, const int pad_y) {
    const int n_channels=C;
    const int input_y=X;
    const int input_x=Y;

    const int input_y_pad=input_y + 2*pad_y;
    const int input_x_pad=input_x + 2*pad_x;
    int size_channel=input_y*input_x;
    int n=n_channels*(input_y_pad*input_x_pad);
    
    float* data_to_send = new float[n]; //Creating piece of memory that will use the simulator
    //Adding y padding

    for(int i=0; i<n; i++) {
	data_to_send[i]=0.0;
    }
    for(int i=0; i<n_channels; i++) {
        for(int y=0; y<input_y; y++) {
            for(int x=0; x<input_x; x++) {
                data_to_send[(n_channels*((y+pad_y)*input_x_pad+x+pad_x)) + i]=bottom_data[i*size_channel + y*(input_x) + x];
            }
        }
    }	

  
    return data_to_send;

}

float* Transform_Filters_Memory (const float* weights, const int K, const int G, const int C, const int R, const int S) {
    
    const int n_channels=C / G;
    const int kernel_y=R;
    const int kernel_x=S;
    const int n_filters=K;    //this->num_output_;
    int size_channel=kernel_y*kernel_x;
    int size_filter=size_channel*n_channels;
    int n=size_filter*n_filters;

    float* filters_to_send = new float[n]; //Creating piece of memory that will use the simulator
    for(int n_f=0; n_f < n_filters; n_f++) {
        for(int i=0; i<n_channels; i++) {
            for(int y=0; y<kernel_y; y++) {
                for(int x=0; x<kernel_x; x++) {
                    filters_to_send[n_f*size_filter+(n_channels*(y*kernel_x+x)) + i]=weights[n_f*size_filter+i*size_channel + y*kernel_x + x];
                }
            }
        }
    }


    return filters_to_send;

}


void Transform_Ofmap_Memory (const float* ofmap_data, float* top_data, const int K, const int X_, const int Y_) {
    const int n_channels=K; //n_filters
    const int output_y=X_;
    const int output_x=Y_;

    int size_channel=output_y*output_x;
    int n=n_channels*size_channel;
    for(int i=0; i<n_channels; i++) {
        for(int y=0; y<output_y; y++) {
            for(int x=0; x<output_x; x++) {
                //data_to_send[(n_channels*(y*input_x+x)) + i]=bottom_data[i*size_channel + y*input_x + x];
                top_data[i*size_channel+y*output_x+x]=ofmap_data[(n_channels*(y*output_x+x)) + i]; //Filling top_data
            }
        }
    }


}





void simulateDenseConvForward(std::string layer_name, float* input, float* weight, float* output, int R, int S, int C, int K, int G, int N, int X, int Y, int X_, int Y_, int strides, int pad_x, int pad_y, std::string path_to_tile, Config stonne_cfg) {
  //Modifying layer name to avoid / characters
  /* const string fixed_layer_name= this->layer_param_.name();
   string layer_name="";
   for(int i=0; i<fixed_layer_name.length(); i++) {
       if (fixed_layer_name[i]=='/') {
           layer_name+="_"; // _ character is changed by /
       }

      else {
          layer_name+=fixed_layer_name[i];
      }
   }
*/
   //Updating X and Y with pad values
   //const int pad_y=this->pad_.cpu_data()[0]; //alto
   const int ifmap_size=C*((X+2*pad_x)*(Y+2*pad_y));
   const int ofmap_size = K*X_*Y_; //X_ and Y_ include padding
   std::cout << "Executing layer " << layer_name << std::endl;
   if(path_to_tile == "") {
	   std::cout << "Tile file parameters must be specified" << std::endl;
	   exit(1);
   }

   //Loading the tile
   Tile tile(path_to_tile);


   float* ifmap_to_send=Transform_Ifmap_Memory(input, C, X, Y, pad_x, pad_y) ;
   float* filters_to_send=Transform_Filters_Memory(weight, K, G, C, R, S);
   float* ofmap_raw = new float[ofmap_size];


   //Tile parameters
   unsigned int T_R = tile.get_T_R();
   unsigned int T_S = tile.get_T_S();
   unsigned int T_C = tile.get_T_C();
   unsigned int T_K = tile.get_T_K();
   unsigned int T_G = tile.get_T_G();
   unsigned int T_N = tile.get_T_N();
   unsigned int T_X_ = tile.get_T_X_();
   unsigned int T_Y_ = tile.get_T_Y_();


   //Executing the accelerator
   Stonne* stonne_instance = new Stonne(stonne_cfg);
   stonne_instance->loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X+2*pad_x, Y+2*pad_y, strides, (address_t) ifmap_to_send, (address_t)filters_to_send, (address_t)ofmap_raw, CNN_DATAFLOW);
   stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
   stonne_instance->run(); //Running the accelerator and generates the output in ofmap_raw
   //sequential_layer(R, S, C, K, G, N, X, Y, strides, (address_t)ifmap_to_send, (address_t)filters_to_send, (address_t)ofmap_raw);

   Transform_Ofmap_Memory(ofmap_raw, output, K, X_, Y_); // Transform simulator memory format to caffe format.     

   //Deleting objects
   delete[] ofmap_raw;
   delete[] ifmap_to_send;
   delete[] filters_to_send;
   delete stonne_instance;

}



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
                DLTensor* input = args[15];
                DLTensor* weight = args[16];
                DLTensor* ouput = args[16];

                //Creating config  to find out if we are going to
                // run a dense or sparse simulation
                Config stonne_config;
                if (path_to_arch_file != "")
                {
                    stonne_config.loadFile(path_to_arch_file);
                }
                
                float* input_raw = (float*)input->data;
                float* weight_raw = (float*)weight->data;
                float* output_raw = (float*)ouput->data;

                std::string layer_name = "TestConv2dLayer";
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
