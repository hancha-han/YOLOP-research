import torch

from lib.config import cfg
from lib.models import get_net


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# YOLOP
# ==================================================
model = get_net(cfg).to(device)

checkpoint = torch.load(
    "weights/End-to-end.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()

print("YOLOP loaded.")


# ==================================================
# 入力
# ==================================================
x = torch.randn(
    1,
    3,
    640,
    640,
    device=device
)


# ==================================================
# F10, F14, F16取得
# ==================================================
original_features = {}


def make_hook(index):

    def hook(module, inputs, output):

        original_features[index] = (
            output.detach().clone()
        )

    return hook


hooks = []

for index in [10, 14, 16]:

    hooks.append(
        model.model[index]
        .register_forward_hook(
            make_hook(index)
        )
    )


# ==================================================
# 通常推論
# ==================================================
with torch.no_grad():

    normal_output = model(x)


for h in hooks:
    h.remove()


normal_detection = normal_output[0]
normal_drivable = normal_output[1]


# ==================================================
# FP16通信をシミュレーション
#
# 車両:
# FP32 -> FP16
#
# 通信
#
# RSU:
# FP16 -> FP32
# ==================================================
fp16_features = {}

for index in [10, 14, 16]:

    transmitted = (
        original_features[index]
        .half()
    )

    fp16_features[index] = (
        transmitted.float()
    )


# ==================================================
# INT8量子化
# ==================================================
int8_features = {}

int8_scales = {}


for index in [10, 14, 16]:

    feature = original_features[index]

    max_value = (
        feature.abs()
        .max()
        .item()
    )

    if max_value == 0:

        scale = 1.0

    else:

        scale = (
            max_value / 127.0
        )


    # 車両側でINT8化
    quantized = torch.clamp(
        torch.round(
            feature / scale
        ),
        -127,
        127
    ).to(
        torch.int8
    )


    # 通信後
    # RSU側でFP32へ復元
    dequantized = (
        quantized.float()
        * scale
    )


    int8_features[index] = (
        dequantized
    )

    int8_scales[index] = (
        scale
    )


# ==================================================
# RSU側処理
# ==================================================
def run_rsu(features):

    cache = [
        None
    ] * len(model.model)

    cache[10] = features[10]
    cache[14] = features[14]
    cache[16] = features[16]

    server_x = features[16]

    detection = None
    drivable_raw = None


    with torch.no_grad():

        for i in range(
            17,
            34
        ):

            block = model.model[i]


            if block.from_ != -1:

                if isinstance(
                    block.from_,
                    int
                ):

                    server_x = (
                        cache[
                            block.from_
                        ]
                    )

                else:

                    server_x = [

                        server_x
                        if j == -1
                        else cache[j]

                        for j in block.from_
                    ]


            server_x = block(
                server_x
            )


            if i == 24:

                detection = (
                    server_x
                )


            if i == 33:

                drivable_raw = (
                    server_x
                )


            cache[i] = server_x


    drivable = torch.sigmoid(
        drivable_raw
    )


    return (
        detection,
        drivable
    )


# ==================================================
# FP16 / INT8でRSU処理
# ==================================================
fp16_detection, fp16_drivable = (
    run_rsu(
        fp16_features
    )
)

int8_detection, int8_drivable = (
    run_rsu(
        int8_features
    )
)


# ==================================================
# DetectionのメインTensorを取得
# ==================================================
def get_detection_tensor(output):

    if isinstance(
        output,
        tuple
    ):

        return output[0]

    return output


normal_det_tensor = (
    get_detection_tensor(
        normal_detection
    )
)

fp16_det_tensor = (
    get_detection_tensor(
        fp16_detection
    )
)

int8_det_tensor = (
    get_detection_tensor(
        int8_detection
    )
)


# ==================================================
# 差を計算
# ==================================================
def max_difference(a, b):

    return torch.max(
        torch.abs(
            a - b
        )
    ).item()


print()
print(
    "===== Output difference ====="
)

print()
print(
    "FP16"
)

print(
    "Detection:",
    max_difference(
        normal_det_tensor,
        fp16_det_tensor
    )
)

print(
    "Drivable Area:",
    max_difference(
        normal_drivable,
        fp16_drivable
    )
)


print()
print(
    "INT8"
)

print(
    "Detection:",
    max_difference(
        normal_det_tensor,
        int8_det_tensor
    )
)

print(
    "Drivable Area:",
    max_difference(
        normal_drivable,
        int8_drivable
    )
)


# ==================================================
# 通信量
# ==================================================
total_elements = sum(
    original_features[i].numel()
    for i in [10, 14, 16]
)


fp32_bytes = (
    total_elements * 4
)

fp16_bytes = (
    total_elements * 2
)

int8_bytes = (
    total_elements
)


print()
print(
    "===== Transmission size ====="
)

print(
    "FP32:",
    fp32_bytes
    / 1024
    / 1024,
    "MiB"
)

print(
    "FP16:",
    fp16_bytes
    / 1024
    / 1024,
    "MiB"
)

print(
    "INT8:",
    int8_bytes
    / 1024
    / 1024,
    "MiB"
)


print()
print(
    "Reduction vs FP32"
)

print(
    "FP16:",
    (
        1
        - fp16_bytes
        / fp32_bytes
    )
    * 100,
    "%"
)

print(
    "INT8:",
    (
        1
        - int8_bytes
        / fp32_bytes
    )
    * 100,
    "%"
)


print()
print(
    "INT8 scales:"
)

for index in [
    10,
    14,
    16
]:

    print(
        f"F{index}:",
        int8_scales[index]
    )
