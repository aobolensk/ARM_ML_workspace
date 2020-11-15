import os
import re
import sys

import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


class Layer:
    def __str__(self) -> str:
        return str(self.__dict__)


def extract_layers(deploy, cmodel):
    caffe.set_mode_gpu()
    caffenet = caffe.Net(deploy, cmodel, caffe.TEST)

    params = caffenet.params.keys()
    source_params = {pr: (caffenet.params[pr][0].data, caffenet.params[pr][1].data) for pr in params}
    layers = dict()

    for name, blobs in caffenet.params.items():
        layer = Layer()
        layer.name = name
        for i in range(len(blobs)):
            # i == 0: Weights
            # i == 1: Bias

            if i == 0:
                outname = name + "_w"
            elif i == 1:
                outname = name + "_b"
            else:
                break

            varname = outname
            if os.path.sep in varname:
                varname = varname.replace(os.path.sep, '_')
            if i == 0:
                layer.shape = list(blobs[i].shape)

            if i == 0:
                layer.weights_file = varname
            elif i == 1:
                layer.bias_file = varname
            np.save(varname, blobs[i].data)
        layers[name] = layer

    parsible_net = caffe_pb2.NetParameter()
    text_format.Merge(open(deploy).read(), parsible_net)

    index = 0
    for layer in parsible_net.layer:
        if layer.name not in layers.keys():
            layers[layer.name] = Layer()
            layers[layer.name].name = layer.name
        layers[layer.name].index = index
        layers[layer.name].type = layer.type
        index += 1
        if layer.type == 'Convolution':
            kernel = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1
            stride = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1
            pad    = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0
            layers[layer.name].kernel = kernel
            layers[layer.name].stride = stride
            layers[layer.name].pad = pad
            layers[layer.name].group = layer.convolution_param.group
        elif layer.type == 'Pooling':
            kernel = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            pad    = layer.pooling_param.pad
            layers[layer.name].kernel = kernel
            layers[layer.name].stride = stride
            layers[layer.name].pad = pad
        elif layer.type == "InnerProduct":
            r = re.match(r".*num_output: ([0-9]+).*", str(layer.ListFields()).replace('\n', ''))
            layers[layer.name].num_output = r.group(1)
        else:
            print(f">>> {layer.name} {layer.type}")

    for _, layer in layers.items():
        print(layer)

    return layers


def create_code(layers):
    code = ""
    code += "graph << common_params.target\n"
    code += "      << common_params.fast_math_hint\n"
    index = 0
    while index < len(layers):
        layer = None
        for maybe_layer in layers:
            if maybe_layer.index == index:
                layer = maybe_layer
                index += 1
                break
        if layer.type == "Convolution":
            code +=  '      << ConvolutionLayer(\n'
            code += f'          {layer.shape[3]}U, {layer.shape[2]}U, {layer.shape[0]}U,\n'
            code += f'          get_weights_accessor(data_path, "/cnn_data/alexnet_model/{layer.weights_file}.npy", weights_layout),\n'
            code += f'          get_weights_accessor(data_path, "/cnn_data/alexnet_model/{layer.bias_file}.npy"),\n'
            code += f'          PadStrideInfo({layer.stride}, {layer.stride}, {layer.pad}, {layer.pad}){f", {layer.group}" if layer.group > 1 else ""})\n'
            code += f'      .set_name("{layer.name}")\n'
        elif layer.type == "Input":
            code += f'      << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))\n'
        elif layer.type == "ReLU":
            code += f'      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("{layer.name}")\n'
        elif layer.type == "LRN":
            code += f'      << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("{layer.name}")\n'
        elif layer.type == "Pooling":
            code += f'      << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, {layer.kernel}, operation_layout, PadStrideInfo({layer.stride}, {layer.stride}, {layer.pad}, {layer.pad}))).set_name("{layer.name}")\n'
        elif layer.type == "InnerProduct":
            code += f'      << FullyConnectedLayer(\n'
            code += f'        {layer.num_output}U,\n'
            code += f'        get_weights_accessor(data_path, "/cnn_data/alexnet_model/{layer.weights_file}.npy", weights_layout),\n'
            code += f'        get_weights_accessor(data_path, "/cnn_data/alexnet_model/{layer.bias_file}.npy"))\n'
            code += f'    .set_name("{layer.name}")\n'
        elif layer.type == "Softmax":
            code += f'      << SoftmaxLayer().set_name("{layer.name}")\n'
        else:
            # TODO: add rest of the layers
            pass
    code += "      << OutputLayer(get_output_accessor(common_params, 5));"
    return code


def main():
    if len(sys.argv) < 3:
        print("python3 convert_model.py <prototxt> <caffemodel>")
        sys.exit(1)
    layers = extract_layers(sys.argv[1], sys.argv[2])
    code = create_code(layers.values())
    print("-------")
    print(code)
    print("-------")


if __name__ == "__main__":
    main()
