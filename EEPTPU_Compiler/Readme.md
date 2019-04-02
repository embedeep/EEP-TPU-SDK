## EEP-TPU Compiler
*eeptpu_compiler* is the EEP-TPU's compiler, it can convert the caffe prototxt and caffemodel files to our EEP-TPU's binary file. 

The output *eeptpu.bin* is suitable for commercial platform EEP-TPU-N1C8A3, NOT for Free-TPU.

You can use our compiler to quickly evaluate whether your algorithm can be used in EEP-TPU or not.

EEP-TPU compiler supports caffe prototxt and caffemodel, if you use other frameworks, such as tensorflow, mxnet or PyTorch, you can use some tools(such as mmdnn) to convert it to caffe files.

### Usage
Run in x86 platform. ( we tested in ubuntu 18.04 )
```
./eeptpu_compiler -h
Usage: 
    ../../eeptpu_compiler [options]
options:
    --help(-h)              # print this help message
    --prototxt <path>       # caffe prototxt file path
    --caffemodel <path>     # caffe caffemodel file path
    --image <path>          # input image path for this neural work
    --mean <values>         # mean values. Format: float array string, such as "103.94,116.78,123.68"
    --norm <values>         # normalize values. Format: float array string, such as "0.017,0.017,0.017"
    --mean_mode <value>     # 0: first mean then norm(default); 1: first norm then mean.
    --output <folder>       # output eeptpu.bin to folder
```

### Examples
- Lenet: only 1 channel
```
./eeptpu_compiler --prototxt ./lenet.prototxt --caffemodel ./lenet.caffemodel --image ./00001_2.pgm --norm "0.00390625" --output ./
```

- Squeezenet: have 3 channels
```
./eeptpu_compiler --prototxt ./squeezenet_v1.1.prototxt --caffemodel ./squeezenet_v1.1.caffemodel --image ./dog-Husky_248.jpg --mean "104.0, 117.0, 123.0" --norm "1.0,1.0,1.0" --mean_mode 0 --output ./
```

### Supported Layer Set
Currently supported layer set V2.0 to run in EEP-TPU, other unsupported layers can run in CPU. More layers will be supported later.

<img src="https://github.com/embedeep/Free-TPU/blob/master/wiki/eeptpu_layer_set.png">


### Some Notes
- **Softmax** layer : Currently compiler not support softmax layer, which often appear at the last layer of classify neural networks. But this softmax layer doesn't affect the final results. We can still use the non-softmax result to do the sort operation. So, if it is last layer of classify neural networks, just remove the softmax layer in your prototxt file.
- **Interpolation** layer : Layer type is "Interp2", suitable for ICNET.
- **Yolo detection_output** layer : This is the last layer of yolo network, you can calculate it in your application. Our library supports multiple layers output, and is also compitable with open source ncnn framework, you can use the output of our library, and calculate the detection_output layer by ncnn. This opertion also needs modify your prototxt, just following the steps below:
```
> Comment or remove the last layer: 
    #layer {
    #  name: "detection_out"
    #  type: "Yolov3DetectionOutput"
    #  bottom: "conv19"
    #  bottom: "conv20"
    #  top: "detection_out"
    #  ......
    #}
> Add new layer as below: ( The attribute values need to be modified according to original layer )
    layer {
      name: "addresses_output"
      type: "AddressOutput"
      bottom: "conv19"
      bottom: "conv20"
      top: "detection_out"
    }
> By using our library, you will get the output data of 'conv19' and 'conv20'.
> Then you could use ncnn detection_output source code to calculate the final result.
```
