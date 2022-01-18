# -----------------------------------------------------------------------------------------------------------
# Tools for BinaryNet
# Cited by https://github.com/DingKe/pytorch_workplace/tree/master/binary/
# -----------------------------------------------------------------------------------------------------------
from torch.autograd import Function

class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = 0
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply
