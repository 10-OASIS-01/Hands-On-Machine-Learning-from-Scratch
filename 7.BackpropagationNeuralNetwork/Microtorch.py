import random
from Autograd import Value

class Module:
    """
    所有神经网络模块的基类。
    提供了重置梯度和获取参数的方法。
    """
    def zero_grad(self):
        """将所有参数的梯度重置为零。"""
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        """返回模块的所有参数（权重和偏置）。"""
        return []

    def describe(self, indent=0):
        """描述模块的结构和参数。"""
        raise NotImplementedError("Each Module must implement the describe method.")


class Neuron(Module):
    """
    表示神经网络中的单个神经元。
    """

    def __init__(self, num_inputs, use_activation=True):
        # 初始化神经元，随机生成权重并设置偏置。
        # 随机初始化权重，范围在-1到1之间
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        # 初始化偏置为0
        self.bias = Value(0)
        # 是否使用非线性激活函数
        self.use_activation = use_activation

    def __call__(self, inputs):
        # 计算神经元的输出。
        # 计算加权和加偏置
        weighted_inputs = [weight * input_val for weight, input_val in zip(self.weights, inputs)]
        weighted_sum = sum(weighted_inputs) + self.bias
        # 如果启用激活函数，则应用ReLU
        if self.use_activation:
            return weighted_sum.relu()
        else:
            return weighted_sum

    def parameters(self):
        """返回神经元的所有参数（权重和偏置）。"""
        return self.weights + [self.bias]

    def describe(self, indent=0):
        ind = '  ' * indent
        weights = ', '.join(f"{w.data:.4f}" for w in self.weights)
        bias = f"{self.bias.data:.4f}"
        activation = "ReLU" if self.use_activation else "Linear"
        desc = f"{ind}{activation}Neuron:\n"
        desc += f"{ind}  Weights: [{weights}]\n"
        desc += f"{ind}  Bias: {bias}\n"
        return desc


    def __repr__(self):
        activation = "ReLU" if self.use_activation else "Linear"
        return f"{activation}Neuron(num_inputs={len(self.weights)})"


class Layer(Module):
    """
    表示神经网络中的一层，由多个神经元组成。
    """

    def __init__(self, num_inputs, num_neurons, use_activation=True):
        # 初始化层，创建指定数量的神经元。
        self.neurons = [Neuron(num_inputs, use_activation) for _ in range(num_neurons)]

    def __call__(self, inputs):
        # 计算每个神经元的输出
        outputs = [neuron(inputs) for neuron in self.neurons]
        # 如果只有一个神经元，直接返回其输出
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def parameters(self):
        """返回层中所有神经元的参数。"""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def describe(self, indent=0):
        ind = '  ' * indent
        desc = f"{ind}Layer with {len(self.neurons)} Neurons:\n"
        for idx, neuron in enumerate(self.neurons, 1):
            desc += f"{ind}  Neuron {idx}: "
            desc += neuron.describe(indent + 2)
        return desc

    def __repr__(self):
        neuron_reprs = ', '.join(str(neuron) for neuron in self.neurons)
        return f"Layer([{neuron_reprs}])"
