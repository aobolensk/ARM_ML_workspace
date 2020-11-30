#!/usr/bin/env python3
import os
import re
import sys

import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


layers = dict()

def get_layer_by_index(index):
    layer = None
    for l in layers.values():
        if l.index == index:
            layer = l
            break
    return layer

def get_parent_dependency_layer(layer_name):
    layer = None
    for l in layers.values():
        if layer_name in l.dependant_layers:
            layer = l
            break
    for l in layers.values():
        if layer_name == l.name:
            layer = l
            break
    return layer


class Layer:
    def __str__(self) -> str:
        return str(self.__dict__)

    is_printed = False

    def get_code(self, folder_name, fork=None):
        code = ""
        if self.is_printed:
            return code
        self.is_printed = True
        # print(layer.top, layer.name, layer.bottom)
        if self.type == "Convolution":
            code +=  '      << ConvolutionLayer(\n'
            code += f'          {self.shape[3]}U, {self.shape[2]}U, {self.shape[0]}U,\n'
            code += f'          get_weights_accessor(data_path, "/cnn_data/{folder_name}/{self.weights_file}.npy", weights_layout),\n'
            code += f'          get_weights_accessor(data_path, "/cnn_data/{folder_name}/{self.bias_file}.npy"),\n'
            code += f'          PadStrideInfo({self.stride}, {self.stride}, {self.pad}, {self.pad}){f", {self.group}" if self.group > 1 else ""})\n'
            code += f'      .set_name("{self.name}")\n'
        elif self.type == "Input":
            pass
        elif self.type == "ReLU":
            code += f'      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("{self.name}")\n'
        elif self.type == "LRN":
            code += f'      << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("{self.name}")\n'
        elif self.type == "Pooling":
            pool = self.pool
            if pool == 0:
                pool = "MAX"
                code += f'      << PoolingLayer(PoolingLayerInfo(PoolingType::{pool}, {self.kernel}, operation_layout, PadStrideInfo({self.stride}, {self.stride}, {self.pad}, {self.pad}))).set_name("{self.name}")\n'
            elif pool == 1:
                pool = "AVG"
                code += f'      << PoolingLayer(PoolingLayerInfo(PoolingType::{pool}, operation_layout)).set_name("{self.name}")\n'
        elif self.type == "InnerProduct":
            code += f'      << FullyConnectedLayer(\n'
            code += f'        {self.num_output}U,\n'
            code += f'        get_weights_accessor(data_path, "/cnn_data/{folder_name}/{self.weights_file}.npy", weights_layout),\n'
            code += f'        get_weights_accessor(data_path, "/cnn_data/{folder_name}/{self.bias_file}.npy"))\n'
            code += f'    .set_name("{self.name}")\n'
        elif self.type == "Softmax":
            code += f'      << SoftmaxLayer().set_name("{self.name}")\n'
        elif self.type == "Concat":
            code += f'      << ConcatLayer(std::move(left_{fork}), std::move(right_{fork})).set_name("{self.name}")\n'
        else:
            # TODO: add rest of the layers
            print(f"WARN: Unknown layer type: {self.type}")
            pass
        return code[:-1]

    def fill(self, layer):
        self.type = layer.type
        self.top = layer.top
        self.bottom = layer.bottom
        self.dependant_layers = []
        if self.top == self.bottom and self.bottom != self.name:
            layers[self.top[0]].dependant_layers.append(self.name)
        if layer.type == 'Convolution':
            self.kernel = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1
            self.stride = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1
            self.pad = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0
            self.group = layer.convolution_param.group
        elif layer.type == 'Pooling':
            self.kernel = layer.pooling_param.kernel_size
            self.stride = layer.pooling_param.stride
            self.pad = layer.pooling_param.pad
            self.pool = layer.pooling_param.pool
        elif layer.type == 'Concat':
            pass
        elif layer.type == 'ReLU':
            pass
        elif layer.type == 'Dropout':
            pass
        elif layer.type == "InnerProduct":
            r = re.match(r".*num_output: ([0-9]+).*", str(layer.ListFields()).replace('\n', ''))
            layers[layer.name].num_output = r.group(1)
        else:
            # TODO: add rest of the layers
            print(f">>> {layer.name} {layer.type}")


def extract_layers(deploy, cmodel):
    caffe.set_mode_gpu()
    caffenet = caffe.Net(deploy, cmodel, caffe.TEST)

    params = caffenet.params.keys()
    source_params = {pr: (caffenet.params[pr][0].data, caffenet.params[pr][1].data) for pr in params}

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
        layers[layer.name].fill(layer)
        index += 1

    return layers


def create_code(folder_name):
    code = ""
    code += "graph << common_params.target\n"
    code += "      << common_params.fast_math_hint\n"
    code += "      << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))"
    fork_starts = []
    fork_ends = []
    for index in range(len(layers.values())):
        layer = get_layer_by_index(index)
        if layer.type == "Concat":
            arr1 = []
            arr2 = []
            iter1 = get_parent_dependency_layer(layer.bottom[0])
            iter2 = get_parent_dependency_layer(layer.bottom[1])
            fork_ends.append(layer.name)
            while True:
                arr1.append(iter1)
                arr2.append(iter2)
                if iter1 in arr2:
                    fork_starts.append(iter1.name)
                    break
                elif iter2 in arr1:
                    fork_starts.append(iter2.name)
                    break
                else: # continue traversing
                    iter1 = layers[iter1.bottom[0]]
                    iter2 = layers[iter2.bottom[0]]
        # code += layer.get_code()
    print(list(zip(fork_starts, fork_ends)))
    forks_index = 0
    resolving_fork_state = 0
    after_fork_count = 0
    for index in range(len(layers.values())):
        layer = get_layer_by_index(index)
        if layer.is_printed:
            continue
        if forks_index < len(fork_ends) and layer.name == fork_ends[forks_index]:
            resolving_fork_state = 0
            forks_index += 1
            code += ";\n"
            code += "graph "
        code += '\n'
        code += layer.get_code(folder_name, fork=forks_index-1)
        after_fork_count += 1
        if resolving_fork_state > 0:
            for dep_name in layer.dependant_layers:
                code += '\n'
                code += layers[dep_name].get_code(folder_name)
        if after_fork_count and resolving_fork_state == 1 and layer.bottom[0] == fork_starts[forks_index]:
            resolving_fork_state = 2
            code += ";\n"
            code += f"SubStream right_{forks_index}(graph);\n"
            code += f"right_{forks_index} "
        elif forks_index < len(fork_starts) and layer.name == fork_starts[forks_index]:
            for dep_name in layer.dependant_layers:
                code += layers[dep_name].get_code(folder_name)
            resolving_fork_state = 1
            code += ";\n"
            code += f"SubStream left_{forks_index}(graph);\n"
            code += f"left_{forks_index} "

    code += '\n'
    code += "      << OutputLayer(get_output_accessor(common_params, 5));"
    return code


def main():
    if len(sys.argv) < 3:
        print("python3 convert_model.py <prototxt> <caffemodel>")
        sys.exit(1)
    layers = extract_layers(sys.argv[1], sys.argv[2])
    name = "alexnet_model"
    code = create_code(name)
    print("-------")
    print(code)
    print("-------")


if __name__ == "__main__":
    main()
