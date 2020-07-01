# TensorRT

Training a RetinaNet model using NGC, Open Images, 

## Issues

Tuesday, 30 June 2020

Training the model is done. 

The link, "Speeding Up..." , has somehow been removed? Currently, waiting for Nvidia's answer about the issue.

Added a preprocessign step from [AastaNV/TRT_object_detection](https://github.com/AastaNV/TRT_object_detection/blob/master/main.py); however, triggers a core dumped error.

Wednesday, 1 July 2020

Somehome, the link is restored overnight.

I cannot proceed further due to this error below 

```
[TensorRT] ERROR: ../rtSafe/cuda/cudaActivationRunner.cpp (103) - Cudnn Error in execute: 3 (CUDNN_STATUS_BAD_PARAM)
[TensorRT] ERROR: FAILED_EXECUTION: std::exception
[TensorRT] ERROR: engine.cpp (179) - Cuda Error in ~ExecutionContext: 700 (an illegal memory access was encountered)
[TensorRT] ERROR: INTERNAL_ERROR: std::exception
[TensorRT] ERROR: Parameter check failed at: ../rtSafe/safeContext.cpp::terminateCommonContext::155, condition: cudnnDestroy(context.cudnn) failure.
[TensorRT] ERROR: Parameter check failed at: ../rtSafe/safeContext.cpp::terminateCommonContext::165, condition: cudaEventDestroy(context.start) failure.
[TensorRT] ERROR: Parameter check failed at: ../rtSafe/safeContext.cpp::terminateCommonContext::170, condition: cudaEventDestroy(context.stop) failure.
[TensorRT] ERROR: ../rtSafe/safeRuntime.cpp (32) - Cuda Error in free: 700 (an illegal memory access was encountered)
terminate called after throwing an instance of 'nvinfer1::CudaError'
what(): std::exception
Aborted (core dumped)
```

## Reference
 - [Speeding Up Deep Learning Inference Using Tensorflow Onnx and TensorRT](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)
 - [Building a Real-time Redaction App Using NVIDIA DeepStream, Part 1: Training](https://developer.nvidia.com/blog/real-time-redaction-app-nvidia-deepstream-part-1-training/)
 - [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
