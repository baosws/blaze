import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inspect

import pytest
import torch
import torch.nn as nn

import blaze as bl


def test_demo_example():
    def forward(x, in_size=10, out_size=20):
        linear = bl.Linear(in_size, out_size)
        return linear(x)

    model = bl.transform(forward, in_size=10, out_size=20)
    model.init(torch.randn(5, 10))
    output = model(torch.randn(5, 10))
    assert output.shape == torch.Size([5, 20])


# ---------- Multi-layer naming ----------

def test_multi_layer_naming():
    """Two Linear calls produce 'linear' and 'linear_1'."""

    def forward(x):
        x = bl.Linear(10, 32)(x)
        x = bl.ReLU()(x)
        x = bl.Linear(32, 5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(4, 10))

    keys = sorted(model._registry.keys())
    assert keys == ["linear", "linear_1", "relu"]

    out = model(torch.randn(4, 10))
    assert out.shape == (4, 5)


# ---------- Nested bl.Module ----------


def test_nested_module():
    class Block(bl.Module):
        def __call__(self, x, dim=32):
            x = bl.Linear(x.shape[-1], dim)(x)
            x = bl.ReLU()(x)
            return x

    def forward(x):
        b1 = Block()
        b2 = Block()
        x = b1(x, dim=64)
        x = b2(x, dim=10)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 16))

    keys = sorted(model._registry.keys())
    assert "block/linear" in keys
    assert "block/relu" in keys
    assert "block_1/linear" in keys
    assert "block_1/relu" in keys

    out = model(torch.randn(2, 16))
    assert out.shape == (2, 10)


# ---------- Parameters work with optimizer ----------


def test_optimizer_integration():
    def forward(x):
        x = bl.Linear(10, 20)(x)
        x = bl.Linear(20, 5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    params = list(model.parameters())
    assert len(params) == 4  # 2 weights + 2 biases

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(torch.randn(2, 10))
    loss = out.sum()
    loss.backward()
    optimizer.step()


# ---------- state_dict / load_state_dict ----------


def test_state_dict_roundtrip():
    def forward(x):
        return bl.Linear(10, 5)(x)

    model = bl.transform(forward)
    model.init(torch.randn(1, 10))

    sd = model.state_dict()
    assert any("weight" in k for k in sd)
    assert any("bias" in k for k in sd)

    # Load into a fresh model
    model2 = bl.transform(forward)
    model2.init(torch.randn(1, 10))
    model2.load_state_dict(sd)

    x = torch.randn(1, 10)
    torch.testing.assert_close(model(x), model2(x))


# ---------- train/eval propagation ----------


def test_train_eval_propagation():
    def forward(x):
        x = bl.Linear(10, 10)(x)
        x = bl.Dropout(0.5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    model.eval()
    x = torch.randn(100, 10)
    out1 = model(x)
    out2 = model(x)
    torch.testing.assert_close(out1, out2)  # deterministic in eval

    model.train()
    # In train mode, dropout introduces randomness (usually)
    # Just check it runs without error
    _ = model(x)


# ---------- BatchNorm stateful ----------


def test_batchnorm():
    def forward(x):
        x = bl.Linear(10, 20)(x)
        x = bl.BatchNorm1d(20)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(4, 10))

    model.train()
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 20)

    model.eval()
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 20)


# ---------- get_parameter ----------


def test_get_parameter():
    def forward(x):
        scale = bl.get_parameter("scale", (1,), init_fn=torch.ones)
        bias = bl.get_parameter("bias", (1,), init_fn=torch.zeros)
        return x * scale + bias

    model = bl.transform(forward)
    model.init(torch.randn(3, 5))

    params = list(model.parameters())
    assert len(params) == 2

    out = model(torch.randn(3, 5))
    assert out.shape == (3, 5)


# ---------- get_state ----------


def test_get_state_basic():
    def forward(x):
        buf = bl.get_state("running_mean", (x.shape[-1],), init_fn=torch.zeros)
        return x + buf

    model = bl.transform(forward)
    model.init(torch.randn(3, 5))

    # Buffer should be in registry and not in parameters
    assert "running_mean" in model._registry
    assert len(list(model.parameters())) == 0

    out = model(torch.randn(3, 5))
    assert out.shape == (3, 5)


def test_get_state_not_trainable():
    def forward(x):
        bl.get_state("buf", (4,))
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 4))

    params = list(model.parameters())
    buffers = list(model.buffers())
    assert len(params) == 0
    assert len(buffers) == 1


def test_get_state_init_fn():
    def forward(x):
        offset = bl.get_state("offset", (3,), init_fn=torch.ones)
        return x + offset

    model = bl.transform(forward)
    model.init(torch.zeros(1, 3))

    x = torch.zeros(1, 3)
    out = model(x)
    torch.testing.assert_close(out, torch.ones(1, 3))


def test_get_state_scoped():
    class Counter(bl.Module):
        def __call__(self, x):
            bl.get_state("count", (1,), init_fn=torch.zeros)
            return x

    def forward(x):
        return Counter()(x)

    model = bl.transform(forward)
    model.init(torch.randn(2, 4))

    assert "counter/count" in model._registry


# ---------- Custom name ----------


def test_custom_name():
    def forward(x):
        x = bl.Linear(10, 10, name="encoder")(x)
        x = bl.Linear(10, 5, name="decoder")(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(1, 10))

    keys = sorted(model._registry.keys())
    assert keys == ["decoder", "encoder"]


# ---------- Conv2d ----------


def test_conv2d():
    def forward(x):
        x = bl.Conv2d(3, 16, kernel_size=3, padding=1)(x)
        x = bl.BatchNorm2d(16)(x)
        x = bl.ReLU()(x)
        x = bl.MaxPool2d(2)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 3, 32, 32))

    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 16, 16)


# ---------- Error: not compiled ----------


def test_error_not_compiled():
    def forward(x):
        return bl.Linear(10, 5)(x)

    model = bl.transform(forward)
    with pytest.raises(RuntimeError, match="not compiled"):
        model(torch.randn(1, 10))


# ---------- Error: outside transform ----------


def test_error_outside_transform():
    with pytest.raises(RuntimeError, match="No active blaze frame"):
        bl.Linear(10, 5)


# ---------- Deeply nested modules ----------


def test_deeply_nested():
    class Inner(bl.Module):
        def __call__(self, x):
            return bl.Linear(x.shape[-1], x.shape[-1])(x)

    class Outer(bl.Module):
        def __call__(self, x):
            x = Inner()(x)
            x = Inner()(x)
            return x

    def forward(x):
        return Outer()(x)

    model = bl.transform(forward)
    model.init(torch.randn(2, 8))

    keys = sorted(model._registry.keys())
    assert "outer/inner/linear" in keys
    assert "outer/inner_1/linear" in keys

    out = model(torch.randn(2, 8))
    assert out.shape == (2, 8)


# ---------- Loop creates distinct modules ----------


def test_loop():
    def forward(x):
        for _ in range(3):
            x = bl.Linear(10, 10)(x)
            x = bl.ReLU()(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(1, 10))

    keys = sorted(model._registry.keys())
    assert "linear" in keys
    assert "linear_1" in keys
    assert "linear_2" in keys
    assert len([k for k in keys if k.startswith("linear")]) == 3

    out = model(torch.randn(1, 10))
    assert out.shape == (1, 10)


# ---------- Weights are actually reused between compile and forward ----------


def test_weights_reused():
    """Verify that compile creates weights and forward reuses them (not new ones)."""

    def forward(x):
        return bl.Linear(10, 5)(x)

    model = bl.transform(forward)
    model.init(torch.randn(1, 10))

    # Get the weight from the registry
    weight_before = model._registry["linear"].weight.data.clone()

    # Forward should use the same module
    x = torch.randn(1, 10)
    out = model(x)
    expected = torch.nn.functional.linear(x, weight_before, model._registry["linear"].bias)
    torch.testing.assert_close(out, expected)

def test_jit():
    """Test that the model can be JIT compiled."""

    def forward(x):
        x = bl.Linear(10, 20)(x)
        x = bl.ReLU()(x)
        x = bl.Linear(20, 5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    jit_model = torch.jit.script(model)
    out = jit_model(torch.randn(2, 10))
    assert out.shape == (2, 5)

def test_jit_training():
    """Test that the model can be JIT compiled and trained."""

    def forward(x):
        x = bl.Linear(10, 20)(x)
        x = bl.ReLU()(x)
        x = bl.Linear(20, 5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    jit_model = torch.jit.script(model)
    optimizer = torch.optim.Adam(jit_model.parameters(), lr=1e-3)

    x = torch.randn(2, 10)
    for _ in range(10):
        optimizer.zero_grad()
        out = jit_model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

def test_jit_multiple_inputs():
    """Test that the model can handle multiple inputs in JIT."""

    def forward(x, y):
        x = bl.Linear(10, 20)(x)
        y = bl.Linear(10, 20)(y)
        z = x + y
        z = bl.ReLU()(z)
        z = bl.Linear(20, 5)(z)
        return z

    model = bl.transform(forward)
    model.init(torch.randn(2, 10), torch.randn(2, 10))

    jit_model = torch.jit.script(model)
    out = jit_model(torch.randn(2, 10), torch.randn(2, 10))
    assert out.shape == (2, 5)

def test_jit_multiple_outputs():
    """Test that the model can handle multiple outputs in JIT."""

    def forward(x):
        x = bl.Linear(10, 20)(x)
        y = bl.Linear(20, 20)(x)
        z = bl.Linear(20, 30)(x)
        return y, z

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    jit_model = torch.jit.script(model)
    y, z = jit_model(torch.randn(2, 10))
    assert y.shape == (2, 20)
    assert z.shape == (2, 30)

def test_multiple_inputs_outputs():
    """Test that the model can handle multiple inputs and outputs."""

    def forward(x, y):
        x = bl.Linear(10, 20)(x)
        y = bl.Linear(10, 20)(y)
        z = x + y
        z = bl.ReLU()(z)
        w = bl.Linear(20, 5)(z)
        v = bl.Linear(20, 5)(z)
        return w, v

    model = bl.transform(forward)
    model.init(torch.randn(2, 10), torch.randn(2, 10))

    out1, out2 = model(torch.randn(2, 10), torch.randn(2, 10))
    assert out1.shape == (2, 5)
    assert out2.shape == (2, 5)

def test_multithreading():
    """Test that multiple threads can use the model without interfering."""

    def forward(x):
        x = bl.Linear(10, 20)(x)
        x = bl.ReLU()(x)
        x = bl.Linear(20, 5)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 10))

    def thread_fn():
        for _ in range(10):
            out = model(torch.randn(2, 10))
            assert out.shape == (2, 5)

    import threading
    threads = [threading.Thread(target=thread_fn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def test_TransformerEncoderLayer():
    """Test that TransformerEncoderLayer can be used."""

    def forward(x):
        x = bl.TransformerEncoderLayer(d_model=10, nhead=2)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 5, 10))
    model = torch.jit.script(model)

    out = model(torch.randn(2, 5, 10))
    assert out.shape == (2, 5, 10)

def test_TransformerEncoder():
    """Test that TransformerEncoder can be used."""

    def forward(x):
        layer = bl.TransformerEncoderLayer(d_model=10, nhead=2)
        x = bl.TransformerEncoder(layer, num_layers=2)(x)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 5, 10))
    model = torch.jit.script(model)

    out = model(torch.randn(2, 5, 10))
    assert out.shape == (2, 5, 10)

def test_TransformerDecoder():
    """Test that TransformerDecoder can be used."""

    def forward(x, memory):
        layer = bl.TransformerDecoderLayer(d_model=10, nhead=2)
        x = bl.TransformerDecoder(layer, num_layers=2)(x, memory)
        return x

    model = bl.transform(forward)
    model.init(torch.randn(2, 5, 10), torch.randn(2, 5, 10))
    model = torch.jit.script(model)

    out = model(torch.randn(2, 5, 10), torch.randn(2, 5, 10))
    assert out.shape == (2, 5, 10)

def test_subclass():
    class PatchedTransformerEncoderLayer(bl.TransformerEncoderLayer):
        def __init__(self, attn_dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attn_weights = None
            self.attn_dropout = attn_dropout

        def _sa_block(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor | None,
            key_padding_mask: torch.Tensor | None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            if attn_mask is not None:
                attn_mask = torch.where(torch.rand_like(attn_mask, dtype=torch.float32) < self.attn_dropout * self.training, torch.zeros_like(attn_mask), attn_mask)
            x, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                is_causal=is_causal,
            )
            self.attn_weights = attn_weights
            return self.dropout1(x)
    
    def forward(x):
        layer = PatchedTransformerEncoderLayer(attn_dropout=0.5, d_model=10, nhead=2, batch_first=True)
        x = layer(x)
        attn_weights = layer.attn_weights
        return x, attn_weights
    
    model = bl.transform(forward)
    model.init(torch.randn(2, 5, 10))
    out = model(torch.randn(2, 5, 10))
    assert out[0].shape == (2, 5, 10)
    assert out[1].shape == (2, 5, 5)
    assert (out[1] >= 0).all() and (out[1] <= 1).all()