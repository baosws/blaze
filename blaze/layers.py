from typing import Optional

import torch.nn as nn

from .context import Mode, get_current_frame
from .utils import camel_to_snake


class _BlazeLayerWrapper:
    _base_name: Optional[str] = None

    def __new__(cls, *args, **kwargs) -> nn.Module:
        frame = get_current_frame()

        name: Optional[str] = kwargs.pop("name", None)
        base = name or cls._base_name or camel_to_snake(cls.__name__)
        local_name = frame.current_counter.next_name(base)

        prefix = frame.current_path()
        full_path = prefix + "/" + local_name if prefix else local_name

        if frame.mode == Mode.INIT:
            nn_cls = next(c for c in cls.__mro__[1:] if issubclass(c, nn.Module))
            module = nn_cls(*args, **kwargs)
            frame.registry[full_path] = module
            frame.call_order.append(full_path)
        else:
            if full_path not in frame.registry:
                raise RuntimeError(
                    f"Module '{full_path}' not found in registry. "
                    f"Function structure changed between init() and forward(). "
                    f"Available: {sorted(frame.registry.keys())}"
                )
            module = frame.registry[full_path]

        return module


class Linear(nn.Linear, _BlazeLayerWrapper): pass
class Bilinear(nn.Bilinear, _BlazeLayerWrapper): pass

# Conv
class Conv1d(nn.Conv1d, _BlazeLayerWrapper): pass
class Conv2d(nn.Conv2d, _BlazeLayerWrapper): pass
class Conv3d(nn.Conv3d, _BlazeLayerWrapper): pass
class ConvTranspose1d(nn.ConvTranspose1d, _BlazeLayerWrapper): pass
class ConvTranspose2d(nn.ConvTranspose2d, _BlazeLayerWrapper): pass
class ConvTranspose3d(nn.ConvTranspose3d, _BlazeLayerWrapper): pass

# Norm
class BatchNorm1d(nn.BatchNorm1d, _BlazeLayerWrapper): pass
class BatchNorm2d(nn.BatchNorm2d, _BlazeLayerWrapper): pass
class BatchNorm3d(nn.BatchNorm3d, _BlazeLayerWrapper): pass
class SyncBatchNorm(nn.SyncBatchNorm, _BlazeLayerWrapper): pass
class InstanceNorm1d(nn.InstanceNorm1d, _BlazeLayerWrapper): pass
class InstanceNorm2d(nn.InstanceNorm2d, _BlazeLayerWrapper): pass
class InstanceNorm3d(nn.InstanceNorm3d, _BlazeLayerWrapper): pass
class LayerNorm(nn.LayerNorm, _BlazeLayerWrapper): pass
class GroupNorm(nn.GroupNorm, _BlazeLayerWrapper): pass
class RMSNorm(nn.RMSNorm, _BlazeLayerWrapper): pass

# Pooling
class MaxPool1d(nn.MaxPool1d, _BlazeLayerWrapper): pass
class MaxPool2d(nn.MaxPool2d, _BlazeLayerWrapper): pass
class MaxPool3d(nn.MaxPool3d, _BlazeLayerWrapper): pass
class AvgPool1d(nn.AvgPool1d, _BlazeLayerWrapper): pass
class AvgPool2d(nn.AvgPool2d, _BlazeLayerWrapper): pass
class AvgPool3d(nn.AvgPool3d, _BlazeLayerWrapper): pass
class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, _BlazeLayerWrapper): pass
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, _BlazeLayerWrapper): pass
class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, _BlazeLayerWrapper): pass
class AdaptiveMaxPool1d(nn.AdaptiveMaxPool1d, _BlazeLayerWrapper): pass
class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d, _BlazeLayerWrapper): pass
class AdaptiveMaxPool3d(nn.AdaptiveMaxPool3d, _BlazeLayerWrapper): pass

# Activation
class ReLU(nn.ReLU, _BlazeLayerWrapper):
    _base_name = "relu"

class ReLU6(nn.ReLU6, _BlazeLayerWrapper): pass
class LeakyReLU(nn.LeakyReLU, _BlazeLayerWrapper): pass
class PReLU(nn.PReLU, _BlazeLayerWrapper): pass
class ELU(nn.ELU, _BlazeLayerWrapper): pass
class SELU(nn.SELU, _BlazeLayerWrapper): pass
class CELU(nn.CELU, _BlazeLayerWrapper): pass
class GELU(nn.GELU, _BlazeLayerWrapper): pass
class Mish(nn.Mish, _BlazeLayerWrapper): pass

class SiLU(nn.SiLU, _BlazeLayerWrapper):
    _base_name = "silu"

class Tanh(nn.Tanh, _BlazeLayerWrapper): pass
class Sigmoid(nn.Sigmoid, _BlazeLayerWrapper): pass
class Hardsigmoid(nn.Hardsigmoid, _BlazeLayerWrapper): pass
class Hardswish(nn.Hardswish, _BlazeLayerWrapper): pass
class Softmax(nn.Softmax, _BlazeLayerWrapper): pass
class LogSoftmax(nn.LogSoftmax, _BlazeLayerWrapper): pass
class Softplus(nn.Softplus, _BlazeLayerWrapper): pass

# Dropout
class Dropout(nn.Dropout, _BlazeLayerWrapper): pass
class Dropout1d(nn.Dropout1d, _BlazeLayerWrapper): pass
class Dropout2d(nn.Dropout2d, _BlazeLayerWrapper): pass
class Dropout3d(nn.Dropout3d, _BlazeLayerWrapper): pass
class AlphaDropout(nn.AlphaDropout, _BlazeLayerWrapper): pass

# Recurrent
class LSTM(nn.LSTM, _BlazeLayerWrapper): pass
class GRU(nn.GRU, _BlazeLayerWrapper): pass
class RNN(nn.RNN, _BlazeLayerWrapper): pass
class LSTMCell(nn.LSTMCell, _BlazeLayerWrapper): pass
class GRUCell(nn.GRUCell, _BlazeLayerWrapper): pass
class RNNCell(nn.RNNCell, _BlazeLayerWrapper): pass

# Embedding
class Embedding(nn.Embedding, _BlazeLayerWrapper): pass
class EmbeddingBag(nn.EmbeddingBag, _BlazeLayerWrapper): pass

# Attention
class MultiheadAttention(nn.MultiheadAttention, _BlazeLayerWrapper): pass

# Shape / spatial
class Flatten(nn.Flatten, _BlazeLayerWrapper): pass
class Unflatten(nn.Unflatten, _BlazeLayerWrapper): pass
class Upsample(nn.Upsample, _BlazeLayerWrapper): pass
class PixelShuffle(nn.PixelShuffle, _BlazeLayerWrapper): pass
class PixelUnshuffle(nn.PixelUnshuffle, _BlazeLayerWrapper): pass

# Misc
class Identity(nn.Identity, _BlazeLayerWrapper): pass

# Transformer
class Transformer(nn.Transformer, _BlazeLayerWrapper): pass
class TransformerEncoder(nn.TransformerEncoder, _BlazeLayerWrapper): pass
class TransformerDecoder(nn.TransformerDecoder, _BlazeLayerWrapper): pass
class TransformerEncoderLayer(nn.TransformerEncoderLayer, _BlazeLayerWrapper): pass
class TransformerDecoderLayer(nn.TransformerDecoderLayer, _BlazeLayerWrapper): pass


__all__ = [
    # Linear
    "Linear", "Bilinear",
    # Conv
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    # Norm
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "LayerNorm", "GroupNorm", "RMSNorm",
    # Pooling
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    # Activation
    "ReLU", "ReLU6", "LeakyReLU", "PReLU",
    "ELU", "SELU", "CELU", "GELU", "Mish", "SiLU",
    "Tanh", "Sigmoid", "Hardsigmoid", "Hardswish",
    "Softmax", "LogSoftmax", "Softplus",
    # Dropout
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
    # Recurrent
    "LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell",
    # Embedding
    "Embedding", "EmbeddingBag",
    # Attention
    "MultiheadAttention",
    # Shape / spatial
    "Flatten", "Unflatten", "Upsample", "PixelShuffle", "PixelUnshuffle",
    # Misc
    "Identity",
    # Transformer
    "Transformer", "TransformerEncoder", "TransformerDecoder",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
]