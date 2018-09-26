#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstdio>
#include <iostream>
#include <typeinfo>

using namespace tensorflow;
using namespace std;

REGISTER_OP("RoiPooling")
.Input("input: float32")
.Input("rois: int32")
.Attr("pool_height: int")
.Attr("pool_width: int")
.Output("output: float32")
.Output("argmax_output: int32");


#define Dtype float

void RoiPoolingKernelLauncher(const float* input, const int* rois, int n_rois, int channels, int height, int width,
                              int pooled_height, int pooled_width, Dtype* output, int* argmax_output);

// IMPORTANT(maciek): need info about storage of the data in memory, assumed something but need the docs confirming it

class RoiPoolingOp : public OpKernel {
    private:
        int pool_height_, pool_width_;
    public:
        explicit RoiPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
                 OP_REQUIRES_OK(context,
                   context->GetAttr("pool_height", &pool_height_));

                 OP_REQUIRES_OK(context,
                   context->GetAttr("pool_width", &pool_width_));
        }


        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            const Tensor& rois_tensor = context->input(1);

            auto input = input_tensor.flat<float>();
            auto rois = rois_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            Tensor* argmax_output_tensor = NULL;

            auto input_shape = input_tensor.shape();
            auto rois_shape = rois_tensor.shape();

            int n_rois = rois_shape.dim_size(0);
            int height = input_shape.dim_size(1);
            int width = input_shape.dim_size(2);
            int channels = input_shape.dim_size(3);

            TensorShape output_shape = TensorShape({static_cast<int64>(n_rois),
                                        static_cast<int64>(channels),
                                        static_cast<int64>(pool_height_),
                                        static_cast<int64>(pool_width_)});

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                        &output_tensor));

            OP_REQUIRES_OK(context, context->allocate_output(1, output_shape,
                        &argmax_output_tensor));

            auto output = output_tensor->template flat<float>();
            auto argmax_output = argmax_output_tensor->template flat<int32>();

            RoiPoolingKernelLauncher(input.data(), rois.data(),
                n_rois, channels,
                height, width,
                pool_height_, pool_width_,
                output.data(), argmax_output.data());
        }
};

REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_GPU), RoiPoolingOp);

///////////// RoiPoolingGrad


REGISTER_OP("RoiPoolingGrad")
.Input("orig_input: float32")
.Input("orig_rois: int32")
.Input("orig_output: float32")
.Input("orig_argmax_output: int32")
.Input("orig_output_grad: float32")
.Attr("pool_height: int")
.Attr("pool_width: int")
.Output("output: float32")
.Doc(R"doc(
 region of interest pooling grad
)doc");

#define Dtype float
void RoiPoolingGradKernelLauncher(const Dtype* orig_input, const int* orig_rois,
                                 int mb_size,
                                 int n_rois, int channels, int height, int width,
                                 int pooled_height, int pooled_width,
                                 const Dtype* orig_output, const int* orig_argmax_output,
                                 const Dtype* orig_output_grad,
                                 Dtype* output);

// IMPORTANT(maciek): need info about storage of the data in memory, assumed something but need the docs confirming it

class RoiPoolingGradOp : public OpKernel {
    private:
        int pool_height_, pool_width_;
    public:
        explicit RoiPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
                 OP_REQUIRES_OK(context,
                   context->GetAttr("pool_height", &pool_height_));

                 OP_REQUIRES_OK(context,
                   context->GetAttr("pool_width", &pool_width_));
        }


        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& orig_input_tensor = context->input(0);
            const Tensor& orig_rois_tensor = context->input(1);
            const Tensor& orig_output_tensor = context->input(2);
            const Tensor& orig_argmax_output_tensor = context->input(3);
            const Tensor& orig_output_grad_tensor = context->input(4);

            auto orig_input = orig_input_tensor.flat<float>();
            auto orig_rois = orig_rois_tensor.flat<int32>();
            auto orig_output = orig_output_tensor.flat<float>();
            auto orig_argmax_output = orig_argmax_output_tensor.flat<int32>();
            auto orig_output_grad = orig_output_grad_tensor.flat<float>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            auto orig_input_shape = orig_input_tensor.shape();
            auto orig_rois_shape = orig_rois_tensor.shape();
            auto grads_shape = orig_input_shape;

            int mb_size = orig_input_shape.dim_size(0);
            int n_rois = orig_rois_shape.dim_size(0);
            int height = orig_input_shape.dim_size(1);
            int width = orig_input_shape.dim_size(2);
            int channels = orig_input_shape.dim_size(3);

            OP_REQUIRES_OK(context, context->allocate_output(0, grads_shape,
                        &output_tensor));

            auto output = output_tensor->template flat<float>();

            // Call the cuda kernel launcher
            RoiPoolingGradKernelLauncher(orig_input.data(), orig_rois.data(),
                mb_size, n_rois, channels, height, width, pool_height_, pool_width_,
                orig_output.data(), orig_argmax_output.data(), orig_output_grad.data(), output.data());
        }
};


REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad").Device(DEVICE_GPU), RoiPoolingGradOp);
