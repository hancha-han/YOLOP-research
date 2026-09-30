import torch

from lib.config import cfg
from lib.models import get_net


# ==================================================
# Device
# ==================================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# YOLOP読み込み
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
# Vehicle側で取り出す特徴量
#
# F10, F14, F16
# ==================================================
vehicle_features = {}


def make_hook(index):

    def hook(module, inputs, output):

        vehicle_features[index] = (
            output.detach().clone()
        )

    return hook


hooks = []

for index in [10, 14, 16]:

    h = model.model[
        index
    ].register_forward_hook(
        make_hook(index)
    )

    hooks.append(h)


# ==================================================
# ① 通常のYOLOP推論
# ==================================================
with torch.no_grad():

    normal_output = model(x)


for h in hooks:
    h.remove()


normal_detection = normal_output[0]
normal_drivable = normal_output[1]


print()
print("===== Vehicle side =====")

for index in [10, 14, 16]:

    print(
        f"F{index}:",
        tuple(
            vehicle_features[index].shape
        )
    )


# ==================================================
# ② RSU側
#
# F10, F14, F16だけを受け取ったところから
# YOLOPの後半を実行
# ==================================================

cache = [None] * len(model.model)

cache[10] = vehicle_features[10]
cache[14] = vehicle_features[14]
cache[16] = vehicle_features[16]


# layer17はF16から開始
server_x = vehicle_features[16]

server_detection = None
server_drivable_raw = None


with torch.no_grad():

    for i in range(
        17,
        34
    ):

        block = model.model[i]


        # ------------------------------------------
        # どの特徴量を入力に使うか
        # ------------------------------------------
        if block.from_ != -1:

            if isinstance(
                block.from_,
                int
            ):

                server_x = cache[
                    block.from_
                ]

            else:

                server_x = [

                    server_x
                    if j == -1
                    else cache[j]

                    for j in block.from_
                ]


        # ------------------------------------------
        # Layer実行
        # ------------------------------------------
        server_x = block(
            server_x
        )


        # ------------------------------------------
        # Detection Head
        # layer24
        # ------------------------------------------
        if i == 24:

            server_detection = (
                server_x
            )


        # ------------------------------------------
        # Drivable Area Head
        # layer33
        # ------------------------------------------
        if i == 33:

            server_drivable_raw = (
                server_x
            )


        # 必要な中間結果を保存
        cache[i] = server_x


# YOLOP本体のforwardでは
# segmentation出力にSigmoidをかけている
server_drivable = torch.sigmoid(
    server_drivable_raw
)


# ==================================================
# 比較用関数
# ==================================================
def compare_tensor(
    name,
    a,
    b
):

    diff = torch.max(
        torch.abs(
            a - b
        )
    ).item()

    print(
        name,
        "max abs difference =",
        diff
    )


print()
print(
    "===== Split inference comparison ====="
)


# ==================================================
# Drivable Area比較
# ==================================================
compare_tensor(
    "Drivable Area",
    normal_drivable,
    server_drivable
)


# ==================================================
# Detection比較
#
# Detectはeval時に
# tupleになっている場合がある
# ==================================================
if isinstance(
    normal_detection,
    tuple
):

    normal_main = (
        normal_detection[0]
    )

else:

    normal_main = (
        normal_detection
    )


if isinstance(
    server_detection,
    tuple
):

    server_main = (
        server_detection[0]
    )

else:

    server_main = (
        server_detection
    )


compare_tensor(
    "Detection",
    normal_main,
    server_main
)


print()
print(
    "If both differences are 0 or extremely close to 0,"
)

print(
    "the Vehicle -> RSU split works."
)
