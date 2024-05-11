import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from .constants import *
# from utils import Profile_Memory

# from utils.my_profiler import *
# from torchvision.models.mobilenetv2 import MobileNetV2Weights


class LinearLayer(nn.Module):

    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def forward(self, x):
        return self.fc(x)

    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    
class FemnistCNN(nn.Module):
   
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.output = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class CIFAR10CNN(nn.Module):
    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    # @calc_exec_time(calc_time=True)
    # @memory_profiler
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        self.rnn.flatten_parameters()                    # 参数压缩
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)                    # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes
    
    return model


def get_mobilenet(n_classes, pretrained=True):
    """
    创建具有 `n_classes` 输出的 MobileNet 模型
    :param n_classes: 类别数目
    :param pretrained: 是否使用预训练权重
    :return: 模型 (nn.Module)
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    
    return model


def get_mobilenet_v3(n_classes, pretrained=True):
    """
    创建具有 `n_classes` 输出的 MobileNet 模型
    :param n_classes: 类别数目
    :param pretrained: 是否使用预训练权重
    :return: 模型 (nn.Module)
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    
    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    
    return model

def get_model(
    name,
    device="cuda",
    seed=666,
    input_dim=None,
    output_dim=None,
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param seed:
    :return: model

    """
    torch.manual_seed(seed)
    if name in SYNTHETIC_LIST:
        if output_dim == 2:
            model = LinearLayer(input_dim, 1).to(device)
        else:
            model = LinearLayer(input_dim, output_dim).to(device)
    elif name in CIFAR10_LIST:
        model = get_mobilenet(n_classes=10).to(device)
    elif name in CIFAR100_LIST:
        model = get_mobilenet(n_classes=100).to(device)
    elif name in EMINST_LERANER_LIST:
        model = FemnistCNN(num_classes=62).to(device)
    elif name in SHAKESPEARE_LIST:
        model = NextCharacterLSTM(
            input_size=SHAKESPEARE_CONFIG["input_size"],
            embed_size=SHAKESPEARE_CONFIG["embed_size"],
            hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
            output_size=SHAKESPEARE_CONFIG["output_size"],
            n_layers=SHAKESPEARE_CONFIG["n_layers"],
        ).to(device)
    else:
        raise NotImplementedError
    return model