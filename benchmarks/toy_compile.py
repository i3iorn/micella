import torch


def f(x):
    return (x * 2 + 1).relu() @ x.t()


gc = torch.compile(f)
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
ok = torch.allclose(gc(x), f(x), atol=1e-2)
print("torch.compile WORKS, correct:", ok)
