from Microtorch import Module, Neuron, Layer
import numpy as np


class MLP(Module):
    """
    多层感知器（MLP）神经网络，用于多分类任务。
    """

    def __init__(self, input_size, layer_sizes, output_size):
        # 将输入尺寸与各层尺寸组合起来
        layer_dimensions = [input_size] + layer_sizes
        self.layers = []

        # 创建每一层
        for i in range(len(layer_sizes)):
            in_size = layer_dimensions[i]
            out_size = layer_dimensions[i + 1]
            # 最后一层通常不使用激活函数
            use_activation = True  # 隐藏层使用激活函数
            layer = Layer(in_size, out_size, use_activation=use_activation)
            self.layers.append(layer)

        # 输出层
        self.output_layer = Layer(layer_sizes[-1], output_size, use_activation=False)

    def __call__(self, inputs):
        # 顺序通过每一层，计算MLP的输出。
        for layer in self.layers:
            inputs = layer(inputs)
        outputs = self.output_layer(inputs)
        return outputs

    def parameters(self):
        """返回MLP中所有层的参数。"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output_layer.parameters())
        return params

    def describe(self, indent=0):
        ind = '  ' * indent
        desc = f"{ind}MLP Structure:\n"
        for idx, layer in enumerate(self.layers, 1):
            desc += f"{ind}  Layer {idx}:\n"
            desc += layer.describe(indent + 2)
        # 描述输出层
        desc += f"{ind}  Output Layer:\n"
        desc += self.output_layer.describe(indent + 2)
        return desc

    def __repr__(self):
        layer_reprs = ', '.join(str(layer) for layer in self.layers)
        layer_reprs += ', ' + str(self.output_layer)
        return f"MLP([{layer_reprs}])"

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # 稳定性处理
    return exps / np.sum(exps, axis=1, keepdims=True)
