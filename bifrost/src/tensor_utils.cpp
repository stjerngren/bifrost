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
                            }
                            else
                            {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}