<div align="center">
  <img src="https://raw.githubusercontent.com/baosws/blaze/main/assets/blaze.png" alt="blaze logo" height="200"/>
</div>

# 🔥 Blaze: Name less. Build more.

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/blaze-torch)](https://pypi.org/project/blaze-torch/)

</div>

A PyTorch adapter inspired by [Haiku's](https://dm-haiku.readthedocs.io) functional programming model. Write stateless forward functions using inline layer calls — no `nn.Module` boilerplate — and get back a proper `nn.Module` with full parameter management and `torch.jit.script` support.

## ✨ Why blaze?

Traditional way to define PyTorch models makes you write every layer **twice** — declared in `__init__`, used in `forward` — and requires naming each one (arch nemesis of programmers), drastically slowing down development. `bl` removes all of that: layers are written once, inline, exactly where they're used.

### Example: Convolutional network

**Traditional PyTorch:**

```python
class ConvNet(nn.Module):
    def __init__(self):          # ← boilerplate you must write
        super().__init__()       # ← boilerplate you must write
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # ← named here...
        self.bn1   = nn.BatchNorm2d(32)               # ← named here...
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # ← named here...
        self.bn2   = nn.BatchNorm2d(64)               # ← named here...
        self.pool  = nn.AdaptiveAvgPool2d(1)          # ← named here...
        self.fc    = nn.Linear(64, 10)                # ← named here...

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # ← ...and used here
        x = F.relu(self.bn2(self.conv2(x)))  # ← ...and used here
        x = self.pool(x).flatten(1)          # ← ...and used here
        return self.fc(x)                    # ← ...and used here, what's the output dim again?

model = ConvNet()
```

**Blaze:**

```python
# No class. No __init__. No self. No invented names. Only logic.
def forward(x):
    x = F.relu(bl.BatchNorm2d(32)(bl.Conv2d(3, 32, 3, padding=1)(x)))
    x = F.relu(bl.BatchNorm2d(64)(bl.Conv2d(32, 64, 3, padding=1)(x)))
    x = bl.AdaptiveAvgPool2d(1)(x).flatten(1)
    return bl.Linear(64, 10)(x)

model = bl.transform(forward)
model.init(torch.randn(1, 3, 32, 32)) # discovers and creates all modules
```

### 🗑️ What gets eliminated

| PyTorch requirement | With blaze |
|---|---|
| `class MyModel(nn.Module)` | Plain function or thin `bl.Module` subclass |
| `def __init__(self)` | Not needed |
| `super().__init__()` | Not needed |
| `self.layer = nn.Linear(...)` | Not needed — layers are created inline |
| Inventing a name for every layer | Auto-derived from class name and deduplicated |
| `nn.ModuleList` / `nn.ModuleDict` for dynamic structure | A plain Python loop or dict |
| Passing hyperparameters through `__init__` to store for `forward` | Just use them directly in the function |

---

## 🚀 Features

- 🧹 **No `nn.Module` boilerplate** — define models as plain functions; layers are called inline.
- 🔌 **Drop-in compatible** — `BlazeModule` is a standard `nn.Module`; training loops, optimizers, `state_dict`, and deployment code need no changes.
- ⚙️ **Automatic parameter management** — weights are discovered on the first `init()` pass and reused on every subsequent forward call.
- 🗂️ **Hierarchical naming** — module paths (e.g. `"block/linear"`, `"block/linear_1"`) are derived automatically from class names and deduplicated per scope.
- 🧩 **Composable modules** — subclass `bl.Module` to build reusable components; scopes nest correctly no matter how deep.
- 🎛️ **Raw parameters** — `get_parameter()` creates a learnable `nn.Parameter` scoped to the current path, without any surrounding module.
- 💾 **Non-trainable state** — `get_state()` creates a buffer tensor (analogous to `hk.get_state`) that is tracked by the module but excluded from gradient updates.
- 🏷️ **Custom names** — any layer call accepts a `name=` keyword to override the auto-derived registry key.
- ⚡ **`torch.jit.script` support** — after `init()`, models can be scripted for deployment with no extra steps.
- 🔄 **Train/eval propagation** — `.train()` / `.eval()` propagate correctly to all registered sub-modules.
- 🧱 **uilt-in layer wrappers** — covers linear, conv, norm, pooling, activation, dropout, recurrent, embedding, attention, transformer, and shape layers.

## 📦 Installation

```bash
pip install blaze-torch
```

## 🧑‍💻 Quickstart
```python
import torch
import blaze as bl

def forward(x):
    x = bl.Linear(10, 64)(x)
    x = bl.ReLU()(x)
    x = bl.Linear(64, 1)(x)
    return x

model = bl.transform(forward)
model.init(torch.randn(4, 10))   # discovers and creates all modules

out = model(torch.randn(4, 10))  # normal nn.Module usage
```

## 📖 Core concepts

### 🔁 Two-phase execution

`bl.transform` wraps your function. Calling `.init(sample_input)` runs an **INIT** pass that discovers every layer call and registers it into an internal registry keyed by its hierarchical path (e.g. `"block/linear"`). Subsequent calls run in **APPLY** mode, reusing the registered modules by call order.

```python
model = bl.transform(forward)
model.init(torch.randn(batch, in_features))  # INIT pass — creates weights
output = model(x)                            # APPLY pass — reuses weights
```

### 🧩 User-defined modules (`bl.Module`)

Subclass `bl.Module` and implement `__call__` to group layers into reusable components. The class name is automatically converted to snake_case for scoping, and repeated instantiations are deduplicated with a numeric suffix.

```python
class MLP(bl.Module):
    def __call__(self, x):
        x = bl.Linear(x.shape[-1], 128)(x)
        x = bl.GELU()(x)
        x = bl.Linear(128, x.shape[-1])(x)
        return x

def forward(x):
    x = MLP()(x)   # parameter names: "mlp/linear", "mlp/gelu", "mlp/linear_1"
    x = MLP()(x)   # parameter names: "mlp_1/linear", ...
    return x
```

### 🎛️ Raw parameters (`bl.get_parameter`)

Create a learnable `nn.Parameter` directly, scoped to the current name context. Analogous to `hk.get_parameter`.

```python
def forward(x):
    scale = bl.get_parameter("scale", (x.shape[-1],), init_fn=torch.ones)
    bias  = bl.get_parameter("bias",  (x.shape[-1],), init_fn=torch.zeros)
    return x * scale + bias
```

### 💾 Non-trainable state (`bl.get_state`)

Create a buffer tensor (non-trainable, tracked by the module). Analogous to `hk.get_state`.

```python
def forward(x):
    running_mean = bl.get_state("running_mean", (x.shape[-1],), init_fn=torch.zeros)
    return x - running_mean
```

### 🏷️ Custom names

Pass `name=` to any layer call to override the auto-derived name:

```python
def forward(x):
    x = bl.Linear(10, 64, name="encoder")(x)
    x = bl.Linear(64, 10, name="decoder")(x)
    return x
```

## ⚡ TorchScript / JIT

After `.init()`, a model can be scripted with `torch.jit.script`:

```python
model = bl.transform(forward)
model.init(torch.randn(2, 10))

scripted = torch.jit.script(model)
out = scripted(torch.randn(2, 10))
```

## 🧱 Available layers

All wrappers accept the same arguments as their `torch.nn` counterparts.

| Category | Layers |
|---|---|
| Linear | `Linear`, `Bilinear` |
| Conv | `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d` |
| Norm | `BatchNorm1d/2d/3d`, `SyncBatchNorm`, `InstanceNorm1d/2d/3d`, `LayerNorm`, `GroupNorm`, `RMSNorm` |
| Pooling | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d` |
| Activation | `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `CELU`, `GELU`, `Mish`, `SiLU`, `Tanh`, `Sigmoid`, `Hardsigmoid`, `Hardswish`, `Softmax`, `LogSoftmax`, `Softplus` |
| Dropout | `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout` |
| Recurrent | `LSTM`, `GRU`, `RNN`, `LSTMCell`, `GRUCell`, `RNNCell` |
| Embedding | `Embedding`, `EmbeddingBag` |
| Attention | `MultiheadAttention` |
| Transformer | `Transformer`, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`, `TransformerDecoderLayer` |
| Shape | `Flatten`, `Unflatten`, `Upsample`, `PixelShuffle`, `PixelUnshuffle` |
| Misc | `Identity` |

## 🏋️ Training

`BlazeModule` is a standard `nn.Module` — use any PyTorch optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

## 🔗 Related projects

| Project | Framework | Description |
|---|---|---|
| [dm-haiku](https://github.com/google-deepmind/dm-haiku) | JAX | The original inspiration. Transforms stateful `hk.Module` code into pure `(init, apply)` function pairs via `hk.transform`. |
| [Flax NNX](https://github.com/google/flax) | JAX | Google's neural network library for JAX. The newer NNX API uses PyTorch-style `__init__`/`__call__` with mutable state; the older Linen API is closer to Haiku's functional style. |
| [Equinox](https://github.com/patrick-kidger/equinox) | JAX | Neural networks as callable PyTrees. Models are plain Python dataclasses; parameters live in the tree rather than a separate registry, making them compatible with `jax.jit`/`jax.grad` directly. |
| [torch.func](https://docs.pytorch.org/docs/stable/func.html) | PyTorch | PyTorch's built-in functional transforms (formerly `functorch`). `torch.func.functional_call` lets you call an existing `nn.Module` with an explicit parameter dict, enabling per-sample gradients, meta-learning, etc. |
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) | PyTorch | Training loop abstraction over `nn.Module`. Reduces boilerplate around the train/val/test cycle but keeps the imperative `nn.Module` programming model. |
