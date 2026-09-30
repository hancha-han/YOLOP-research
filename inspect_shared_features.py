import torch

from lib.config import cfg
from lib.models import get_net


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================
# YOLOP
# ==========================================
model = get_net(cfg).to(device)

checkpoint = torch.load(
    "weights/End-to-end.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()


# ==========================================
# 640 x 640 のダミー画像
# ==========================================
x = torch.randn(
    1,
    3,
    640,
    640,
    device=device
)


# ==========================================
# layer 10, 14, 16 を保存
# ==========================================
features = {}


def make_hook(name):

    def hook(
        module,
        input,
        output
    ):
        features[name] = output.detach()

    return hook


hooks = []

for layer_index in [
    10,
    14,
    16
]:

    hook = model.model[
        layer_index
    ].register_forward_hook(
        make_hook(
            f"layer_{layer_index}"
        )
    )

    hooks.append(hook)


# ==========================================
# Forward
# ==========================================
with torch.no_grad():

    output = model(x)


# ==========================================
# 結果
# ==========================================
print()
print(
    "===== Shared feature candidates ====="
)


total_elements = 0
total_bytes_fp32 = 0


for name in [
    "layer_10",
    "layer_14",
    "layer_16"
]:

    f = features[name]

    elements = f.numel()

    bytes_fp32 = (
        elements
        * 4
    )

    total_elements += elements
    total_bytes_fp32 += bytes_fp32


    print()
    print(name)

    print(
        "shape:",
        tuple(f.shape)
    )

    print(
        "elements:",
        elements
    )

    print(
        "FP32 bytes:",
        bytes_fp32
    )

    print(
        "FP32 MB:",
        bytes_fp32
        / 1024
        / 1024
    )


print()
print(
    "===== TOTAL ====="
)

print(
    "Total elements:",
    total_elements
)

print(
    "FP32 total MB:",
    total_bytes_fp32
    / 1024
    / 1024
)

print(
    "FP16 total MB:",
    total_bytes_fp32
    / 2
    / 1024
    / 1024
)


for hook in hooks:
    hook.remove()
