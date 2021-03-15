#include "STONNEModel.h"
#include "include/Analyzer.h"
#include "include/MAERIModel.h"
#include "include/DNNModel.h"

void mRNA_tiles(Stonne *stonne_instance, int ms_num, int dn_bw, int rn_bw,
int X, int Y, int C, int R, int S, int X_, int Y_, int K, int N, int stride)
{
    std::cout << "Create MAERI" << std::endl;
    mrna::Maeri *maeri = new mrna::Maeri(ms_num, dn_bw, rn_bw);

    std::cout << "Create DNN" << std::endl;
    mrna::DNNModel *dnn = new mrna::DNNModel();
    dnn->cnn_input->input_x = X;
    dnn->cnn_input->input_y = Y;
    dnn->cnn_input->input_channel = C;
    dnn->cnn_input->input_batch = 1;

    dnn->cnn_filter->filter_x = R;
    dnn->cnn_filter->filter_y = S;
    dnn->cnn_filter->filter_channel = 1;
    dnn->cnn_filter->filter_number = K;
    dnn->cnn_filter->window_stride = stride;

    dnn->cnn_output->output_x = X_;
    dnn->cnn_output->output_y = Y_;
    dnn->cnn_output->output_channel = K;
    dnn->cnn_output->output_batch = N;

    dnn->dnn_hidden->hidden_x = 0;
    dnn->dnn_hidden->hidden_y = 0;
    dnn->dnn_hidden->hidden_channel = 0;

    std::cout << "Create analyzer" << std::endl;
    mrna::Analyzer *analyzer = new mrna::Analyzer(maeri, dnn, mrna::performance);

    analyzer->setshowenergy(false);
    // TODO: Suuport other mRNA goals
    analyzer->setoptgoal(mrna::performance);

    mrna::OptGoal opt_goal = mrna::performance;

    mrna::MappingStrategy* bestmap;
    std::cout << "Analyse" << std::endl;
    if (analyzer->dnn_model->layer_type == "CONV")
    {
        bestmap = analyzer->AnalyzeCNN(opt_goal);
    }
    // TODO: ADD FC support
    //else if (analyzer->dnn_model->layer_type == "FC")
    //{
    //    analyzer->AnalyzeFC(Profile_result, opt_goal);
    //}

    std::cout << "CGet best" << std::endl;
    unsigned int T_R = bestmap->kernel_x;
    unsigned int T_S = bestmap->kernel_y;
    unsigned int T_C = bestmap->kernel_c;
    unsigned int T_K = bestmap->kernel_n;
    unsigned int T_G = 1;
    unsigned int T_N = bestmap->kernel_in;
    unsigned int T_X_ = bestmap->kernel_ox;
    unsigned int T_Y_ = bestmap->kernel_oy;
    std::cout << "Load tile" << std::endl;
    stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
}