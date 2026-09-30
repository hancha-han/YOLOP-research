import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img


NUM_IMAGES = 100

IMAGE_DIR = Path(
    "/home/tada/Application/datasets/"
    "bdd100k/100k/val"
)

FEATURE_LAYERS = [
    10,
    14,
    16
]


# ==================================================
# GPU
# ==================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ==================================================
# YOLOP
# ==================================================
model = get_net(
    cfg
).to(device)

checkpoint = torch.load(
    "weights/End-to-end.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()

print(
    "YOLOP loaded."
)


# ==================================================
# 正規化
# YOLOP demoと同じ考え方
# ==================================================
normalize = transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],
    std=[
        0.229,
        0.224,
        0.225
    ]
)


# ==================================================
# 画像読み込み
# ==================================================
def load_image(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not load: {image_path}"
        )


    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # YOLOP用letterbox
    image, _, _ = letterbox_for_img(
        image,
        new_shape=640,
        auto=True
    )


    image = image.transpose(
        2,
        0,
        1
    )

    image = np.ascontiguousarray(
        image
    )


    tensor = torch.from_numpy(
        image
    ).float()

    tensor /= 255.0

    tensor = normalize(
        tensor
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)


    return tensor


# ==================================================
# Feature取得
# ==================================================
features = {}


def make_hook(
    index
):

    def hook(
        module,
        inputs,
        output
    ):

        features[index] = (
            output.detach().clone()
        )

    return hook


hooks = []

for index in FEATURE_LAYERS:

    hooks.append(
        model.model[index]
        .register_forward_hook(
            make_hook(index)
        )
    )


# ==================================================
# RSU側
# ==================================================
def run_rsu(
    feature_dict
):

    cache = [
        None
    ] * len(
        model.model
    )


    cache[10] = (
        feature_dict[10]
    )

    cache[14] = (
        feature_dict[14]
    )

    cache[16] = (
        feature_dict[16]
    )


    x = feature_dict[16]

    detection = None
    drivable_raw = None


    with torch.no_grad():

        for i in range(
            17,
            34
        ):

            block = (
                model.model[i]
            )


            if block.from_ != -1:

                if isinstance(
                    block.from_,
                    int
                ):

                    x = cache[
                        block.from_
                    ]

                else:

                    x = [

                        x
                        if j == -1
                        else cache[j]

                        for j
                        in block.from_
                    ]


            x = block(
                x
            )


            if i == 24:

                detection = x


            if i == 33:

                drivable_raw = x


            cache[i] = x


    drivable = torch.sigmoid(
        drivable_raw
    )


    return (
        detection,
        drivable
    )


# ==================================================
# Detection Tensor取得
# ==================================================
def get_detection_tensor(
    output
):

    if isinstance(
        output,
        tuple
    ):

        return output[0]

    return output


# ==================================================
# FP16化
# ==================================================
def make_fp16_features(
    source
):

    result = {}

    for index in FEATURE_LAYERS:

        # 車両でFP16
        transmitted = (
            source[index]
            .half()
        )

        # RSUでFP32に戻す
        result[index] = (
            transmitted.float()
        )

    return result


# ==================================================
# INT8化
# ==================================================
def make_int8_features(
    source
):

    result = {}


    for index in FEATURE_LAYERS:

        f = source[index]

        max_value = (
            f.abs()
            .max()
            .item()
        )


        if max_value == 0:

            scale = 1.0

        else:

            scale = (
                max_value
                / 127.0
            )


        quantized = torch.clamp(
            torch.round(
                f / scale
            ),
            -127,
            127
        ).to(
            torch.int8
        )


        result[index] = (
            quantized.float()
            * scale
        )


    return result


# ==================================================
# 集計
# ==================================================
image_paths = sorted(
    IMAGE_DIR.glob(
        "*.jpg"
    )
)[:NUM_IMAGES]


fp16_det_max_list = []
fp16_det_mean_list = []

fp16_seg_max_list = []
fp16_seg_mean_list = []

int8_det_max_list = []
int8_det_mean_list = []

int8_seg_max_list = []
int8_seg_mean_list = []

fp16_seg_pixel_agreement = []
int8_seg_pixel_agreement = []


# ==================================================
# 100枚処理
# ==================================================
for number, image_path in enumerate(
    image_paths,
    start=1
):

    features.clear()


    image = load_image(
        image_path
    )


    # ------------------------------------------
    # 元のFP32 YOLOP
    # ------------------------------------------
    with torch.no_grad():

        normal_output = model(
            image
        )


    normal_detection = (
        get_detection_tensor(
            normal_output[0]
        )
    )

    normal_drivable = (
        normal_output[1]
    )


    original_features = {

        index:
            features[index]
            .clone()

        for index
        in FEATURE_LAYERS
    }


    # ------------------------------------------
    # FP16
    # ------------------------------------------
    fp16_features = (
        make_fp16_features(
            original_features
        )
    )

    (
        fp16_detection,
        fp16_drivable

    ) = run_rsu(
        fp16_features
    )

    fp16_detection = (
        get_detection_tensor(
            fp16_detection
        )
    )


    # ------------------------------------------
    # INT8
    # ------------------------------------------
    int8_features = (
        make_int8_features(
            original_features
        )
    )

    (
        int8_detection,
        int8_drivable

    ) = run_rsu(
        int8_features
    )

    int8_detection = (
        get_detection_tensor(
            int8_detection
        )
    )


    # ==================================================
    # Detection差
    # ==================================================
    fp16_det_diff = torch.abs(
        normal_detection
        - fp16_detection
    )

    int8_det_diff = torch.abs(
        normal_detection
        - int8_detection
    )


    fp16_det_max_list.append(
        fp16_det_diff
        .max()
        .item()
    )

    fp16_det_mean_list.append(
        fp16_det_diff
        .mean()
        .item()
    )


    int8_det_max_list.append(
        int8_det_diff
        .max()
        .item()
    )

    int8_det_mean_list.append(
        int8_det_diff
        .mean()
        .item()
    )


    # ==================================================
    # Drivable差
    # ==================================================
    fp16_seg_diff = torch.abs(
        normal_drivable
        - fp16_drivable
    )

    int8_seg_diff = torch.abs(
        normal_drivable
        - int8_drivable
    )


    fp16_seg_max_list.append(
        fp16_seg_diff
        .max()
        .item()
    )

    fp16_seg_mean_list.append(
        fp16_seg_diff
        .mean()
        .item()
    )


    int8_seg_max_list.append(
        int8_seg_diff
        .max()
        .item()
    )

    int8_seg_mean_list.append(
        int8_seg_diff
        .mean()
        .item()
    )


    # ==================================================
    # Drivable最終クラスが
    # FP32と同じか
    # ==================================================
    normal_seg_class = torch.argmax(
        normal_drivable,
        dim=1
    )

    fp16_seg_class = torch.argmax(
        fp16_drivable,
        dim=1
    )

    int8_seg_class = torch.argmax(
        int8_drivable,
        dim=1
    )


    fp16_agreement = (
        normal_seg_class
        == fp16_seg_class
    ).float().mean().item()


    int8_agreement = (
        normal_seg_class
        == int8_seg_class
    ).float().mean().item()


    fp16_seg_pixel_agreement.append(
        fp16_agreement
    )

    int8_seg_pixel_agreement.append(
        int8_agreement
    )


    if (
        number % 10 == 0
        or number == 1
    ):

        print(
            f"Processed "
            f"{number}/{NUM_IMAGES}"
        )


# ==================================================
# Hook削除
# ==================================================
for hook in hooks:

    hook.remove()


# ==================================================
# 平均を出す
# ==================================================
def avg(values):

    return (
        sum(values)
        / len(values)
    )


print()
print(
    "===================================="
)

print(
    "RESULT: 100 REAL BDD100K IMAGES"
)

print(
    "===================================="
)


print()
print(
    "FP16"
)

print(
    "Detection max diff average:",
    avg(
        fp16_det_max_list
    )
)

print(
    "Detection mean diff average:",
    avg(
        fp16_det_mean_list
    )
)

print(
    "Drivable max diff average:",
    avg(
        fp16_seg_max_list
    )
)

print(
    "Drivable mean diff average:",
    avg(
        fp16_seg_mean_list
    )
)

print(
    "Drivable pixel agreement:",
    avg(
        fp16_seg_pixel_agreement
    )
    * 100,
    "%"
)


print()
print(
    "INT8"
)

print(
    "Detection max diff average:",
    avg(
        int8_det_max_list
    )
)

print(
    "Detection mean diff average:",
    avg(
        int8_det_mean_list
    )
)

print(
    "Drivable max diff average:",
    avg(
        int8_seg_max_list
    )
)

print(
    "Drivable mean diff average:",
    avg(
        int8_seg_mean_list
    )
)

print(
    "Drivable pixel agreement:",
    avg(
        int8_seg_pixel_agreement
    )
    * 100,
    "%"
)


# ==================================================
# 通信量
# ==================================================
total_elements = sum(
    original_features[index]
    .numel()

    for index
    in FEATURE_LAYERS
)


print()
print(
    "Transmission size"
)

print(
    "FP32:",
    total_elements
    * 4
    / 1024
    / 1024,
    "MiB"
)

print(
    "FP16:",
    total_elements
    * 2
    / 1024
    / 1024,
    "MiB"
)

print(
    "INT8:",
    total_elements
    / 1024
    / 1024,
    "MiB"
)
