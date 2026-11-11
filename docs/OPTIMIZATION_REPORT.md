# M2 Edge AI Optimization Deep Dive

This technical report details the engineering methods applied to optimize the trained YOLO Nano enclosure detection model for resource-constrained edge computing on the Raspberry Pi 5.

---

## 1. The Optimization Problem at the Edge

Embedded edge platforms like the Raspberry Pi 5 present specific constraints:
- **Memory Bandwidth:** While the BCM2712 Cortex-A76 cores operate at 2.4 GHz, memory bus throughput (~17 GB/s) is shared between the CPU, GPU, and system processes. Loading heavy 32-bit floating point model weights on every frame creates a memory bus bottleneck.
- **RAM Constraints:** System RAM must accommodate OS buffers, GUI desktop environments, and camera frame ring-buffers. PyTorch's dynamic computational graph and autograd machinery consume upwards of 500 MB of system RAM.
- **Thermal Budget:** Sustained high CPU utilization causes thermal throttling if core temperatures exceed 80°C.

---

## 2. Optimization Strategy 1: Open Neural Network Exchange (ONNX) Export & Slimming

The trained PyTorch model (`object-detection-n6b60-2-yolo26n-t2`) was converted to ONNX Opset 17 with `onnxslim` graph optimization:
1. **Operator Fusion:** Consecutive convolution, batch normalization, and activation operations (e.g. `Conv + BatchNorm + SiLU`) were fused into single optimized compute nodes.
2. **Constant Folding:** Static weights, biases, and anchor strides were pre-computed at export time.
3. **Decoupled Detection Head:** Output tensor shape was standardized to `(1, 5, 8400)`, decoupling box regression and classification probabilities for streamlined C++ processing.

---

## 3. Optimization Strategy 2: Dynamic INT8 Quantization

Dynamic quantization was applied using ONNX Runtime's quantization toolkit:
- **Weights Precision:** 32-bit IEEE float $\rightarrow$ 8-bit two's complement integer (`QInt8`).
- **Dynamic Scale Factor:** Activations are dynamically scaled to 8-bit during inference, eliminating the need for a complex calibration dataset while preserving dynamic range.
- **Node Transformation:** Replaced standard FP32 convolution operators with **102 `ConvInteger` nodes** and **92 `DynamicQuantizeLinear` nodes**.

### Quantization Mathematical Impact:
$$\text{Quantized Weight} = \text{round}\left(\frac{\text{FP32 Weight}}{\text{Scale}}\right) + \text{ZeroPoint}$$

- **File Size Reduction:** Reduced from **9.34 MB to 2.72 MB** (70.9% reduction).
- **Memory Bandwidth Reduction:** 4x fewer bytes transferred across the memory bus per inference pass.
- **Accuracy Retention:** mAP@0.50 maintained at **96.60%**, while Precision reached **99.67%**.

---

## 4. Optimization Strategy 3: Multi-Core Thread Allocation & ARM NEON Vector SIMD

- **SIMD Vectorization:** ARM NEON registers (128-bit) process four 32-bit floats or sixteen 8-bit integers in a single CPU cycle.
- **Thread Tuning:** Through `sess_options.intra_op_num_threads = 4`, inference matrix operations are partitioned evenly across all four physical Cortex-A76 cores, maximizing hardware utilization without thread scheduling thrash.
- **Benchmark Latency:** Pure model inference achieves **251.99 ms median** at batch 1 (3.87 FPS pure throughput). Full end-to-end webcam pipeline runs at **~3.2 to 3.8 FPS**.
