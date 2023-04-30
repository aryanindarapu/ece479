/*
 * Copyright (c) 2016-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class NEONCNNExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        // Create memory manager components
        // We need 2 memory managers: 1 for handling the tensors within the functions (mm_layers) and 1 for handling the input and output tensors of the functions (mm_transitions))
        auto lifetime_mgr0  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto lifetime_mgr1  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto pool_mgr0      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto pool_mgr1      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto mm_layers      = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0); // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1); // Create the memory manager

        // The weights and biases tensors should be initialized with the values inferred with the training

        // Set memory manager where allowed to manage internal memory requirements
        conv1   = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv2   = std::make_unique<NEConvolutionLayer>(mm_layers);
        fc3     = std::make_unique<NEFullyConnectedLayer>(mm_layers);
        softmax = std::make_unique<NESoftmaxLayer>(mm_layers);

        /* [Initialize tensors] */

        // Initialize src tensor
        constexpr unsigned int width_src_image  = 32;
        constexpr unsigned int height_src_image = 32;
        constexpr unsigned int ifm_src_img      = 1;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        // Initialize tensors of conv0
        constexpr unsigned int kernel_x_conv1 = 5;
        constexpr unsigned int kernel_y_conv1 = 5;
        constexpr unsigned int ofm_conv1      = 8;

        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, src_shape.z(), ofm_conv1);
        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
        const TensorShape out_shape_conv1(src_shape.x(), src_shape.y(), weights_shape_conv1[3]);

        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        // Initialize tensor of act0
        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        // Initialize tensor of pool0
        TensorShape out_shape_pool1 = out_shape_conv1;
        out_shape_pool1.set(0, out_shape_pool1.x() / 2);
        out_shape_pool1.set(1, out_shape_pool1.y() / 2);
        out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));

        // Initialize tensors of conv1
        constexpr unsigned int kernel_x_conv2 = 3;
        constexpr unsigned int kernel_y_conv2 = 3;
        constexpr unsigned int ofm_conv2      = 16;

        const TensorShape weights_shape_conv2(kernel_x_conv2, kernel_y_conv2, out_shape_pool1.z(), ofm_conv2);

        const TensorShape biases_shape_conv2(weights_shape_conv2[3]);
        const TensorShape out_shape_conv2(out_shape_pool1.x(), out_shape_pool1.y(), weights_shape_conv2[3]);

        weights2.allocator()->init(TensorInfo(weights_shape_conv2, 1, DataType::F32));
        biases2.allocator()->init(TensorInfo(biases_shape_conv2, 1, DataType::F32));
        out_conv2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));

        // Initialize tensor of act1
        out_act2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));

        // Initialize tensor of pool1
        TensorShape out_shape_pool2 = out_shape_conv2;
        out_shape_pool2.set(0, out_shape_pool2.x() / 2);
        out_shape_pool2.set(1, out_shape_pool2.y() / 2);
        out_pool2.allocator()->init(TensorInfo(out_shape_pool2, 1, DataType::F32));

        // Initialize tensor of fc0
        constexpr unsigned int num_labels = 128;

        const TensorShape weights_shape_fc3(out_shape_pool2.x() * out_shape_pool2.y() * out_shape_pool2.z(), num_labels);
        const TensorShape biases_shape_fc3(num_labels);
        const TensorShape out_shape_fc3(num_labels);

        weights3.allocator()->init(TensorInfo(weights_shape_fc3, 1, DataType::F32));
        biases3.allocator()->init(TensorInfo(biases_shape_fc3, 1, DataType::F32));
        out_fc3.allocator()->init(TensorInfo(out_shape_fc3, 1, DataType::F32));

        // Initialize tensor of act2
        out_act3.allocator()->init(TensorInfo(out_shape_fc3, 1, DataType::F32));

        // Initialize tensor of softmax
        const TensorShape out_shape_softmax(out_shape_fc3.x());
        out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

        constexpr auto data_layout = DataLayout::NCHW;

        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] */

        // in:32x32x1: 5x5 convolution, 8 output features maps (OFM)
        conv1->configure(&src, &weights1, &biases1, &out_conv1, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 2 /* pad_x */, 2 /* pad_y */));

        // in:32x32x8, out:32x32x8, Activation function: relu
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:32x32x8, out:16x16x8 (2x2 pooling), Pool type function: Max
        pool1.configure(&out_act1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 2, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        // in:16x16x8: 3x3 convolution, 16 output features maps (OFM)
        conv2->configure(&out_pool1, &weights2, &biases2, &out_conv2, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));

        // in:16x16x16, out:16x16x16, Activation function: relu
        act2.configure(&out_conv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:16x16x16, out:8x8x16 (2x2 pooling), Pool type function: Average
        pool2.configure(&out_act2, &out_pool2, PoolingLayerInfo(PoolingType::AVG, 2, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        // in:8x8x16, out:128
        fc3->configure(&out_pool2, &weights3, &biases3, &out_fc3);

        // in:128, out:128, Activation function: relu
        act3.configure(&out_fc3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        // in:128, out:128
        softmax->configure(&out_act3, &out_softmax);

        /* -----------------------End: [Configure functions] */

        /*[ Add tensors to memory manager ]*/

        // We need 2 memory groups for handling the input and output
        // We call explicitly allocate after manage() in order to avoid overlapping lifetimes
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        memory_group1 = std::make_unique<MemoryGroup>(mm_transitions);

        memory_group0->manage(&out_conv1);
        out_conv1.allocator()->allocate();
        memory_group1->manage(&out_act1);
        out_act1.allocator()->allocate();
        memory_group0->manage(&out_pool1);
        out_pool1.allocator()->allocate();
        memory_group1->manage(&out_conv2);
        out_conv2.allocator()->allocate();
        memory_group0->manage(&out_act2);
        out_act2.allocator()->allocate();
        memory_group1->manage(&out_pool2);
        out_pool2.allocator()->allocate();
        memory_group0->manage(&out_fc3);
        out_fc3.allocator()->allocate();
        memory_group1->manage(&out_act3);
        out_act3.allocator()->allocate();
        memory_group0->manage(&out_softmax);
        out_softmax.allocator()->allocate();

        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights1.allocator()->allocate();
        weights2.allocator()->allocate();
        weights3.allocator()->allocate();
        biases1.allocator()->allocate();
        biases2.allocator()->allocate();
        biases3.allocator()->allocate();

        /* -----------------------End: [Allocate tensors] */

        // Populate the layers manager. (Validity checks, memory allocations etc)
        mm_layers->populate(allocator, 1 /* num_pools */);

        // Populate the transitions manager. (Validity checks, memory allocations etc)
        mm_transitions->populate(allocator, 2 /* num_pools */);

        return true;
    }
    void do_run() override
    {
        // Acquire memory for the memory groups
        memory_group0->acquire();
        memory_group1->acquire();

        conv1->run();
        act1.run();
        pool1.run();
        conv2->run();
        act2.run();
        pool2.run();
        fc3->run();
        act3.run();
        softmax->run();

        // Release memory
        memory_group0->release();
        memory_group1->release();
    }

private:
    // The src tensor should contain the input image
    Tensor src{};

    // Intermediate tensors used
    Tensor weights1{};
    Tensor weights2{};
    Tensor weights3{};
    Tensor biases1{};
    Tensor biases2{};
    Tensor biases3{};
    Tensor out_conv1{};
    Tensor out_conv2{};
    Tensor out_act1{};
    Tensor out_act2{};
    Tensor out_act3{};
    Tensor out_pool1{};
    Tensor out_pool2{};
    Tensor out_fc3{};
    Tensor out_softmax{};

    // Allocator
    Allocator allocator{};

    // Memory groups
    std::unique_ptr<MemoryGroup> memory_group0{};
    std::unique_ptr<MemoryGroup> memory_group1{};

    // Layers
    std::unique_ptr<NEConvolutionLayer>    conv1{};
    std::unique_ptr<NEConvolutionLayer>    conv2{};
    std::unique_ptr<NEFullyConnectedLayer> fc3{};
    std::unique_ptr<NESoftmaxLayer>        softmax{};
    NEPoolingLayer                         pool1{};
    NEPoolingLayer                         pool2{};
    NEActivationLayer                      act1{};
    NEActivationLayer                      act2{};
    NEActivationLayer                      act3{};
};

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCNNExample>(argc, argv);
}
