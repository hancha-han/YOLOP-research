from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img


NUM_IMAGES = 10000

IMAGE_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/100k/val"
)

MASK_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/labels/val"
)

FEATURE_LAYERS = [10, 14, 16]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


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


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def load_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    h0, w0 = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    processed, _, _ = letterbox_for_img(
        image_rgb,
        new_shape=640,
        auto=True
    )

    input_h, input_w = processed.shape[:2]

    r = min(
        640 / h0,
        640 / w0
    )

    resized_w = int(
        round(w0 * r)
    )

    resized_h = int(
        round(h0 * r)
    )

    pad_w = input_w - resized_w
    pad_h = input_h - resized_h

    left = int(
        round(pad_w / 2 - 0.1)
    )

    top = int(
        round(pad_h / 2 - 0.1)
    )

    processed = processed.transpose(
        2, 0, 1
    )

    processed = np.ascontiguousarray(
        processed
    )

    tensor = torch.from_numpy(
        processed
    ).float()

    tensor /= 255.0

    tensor = normalize(
        tensor
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)

    info = {
        "input_h": input_h,
        "input_w": input_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "top": top,
        "left": left
    }

    return tensor, info


def load_gt_mask(mask_path, info):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    gt = (
        mask > 0
    ).astype(
        np.uint8
    )

    gt_resized = cv2.resize(
        gt,
        (
            info["resized_w"],
            info["resized_h"]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    gt_input = np.zeros(
        (
            info["input_h"],
            info["input_w"]
        ),
        dtype=np.uint8
    )

    top = info["top"]
    left = info["left"]

    gt_input[
        top:top + info["resized_h"],
        left:left + info["resized_w"]
    ] = gt_resized

    valid = np.zeros_like(
        gt_input,
        dtype=bool
    )

    valid[
        top:top + info["resized_h"],
        left:left + info["resized_w"]
    ] = True

    return gt_input, valid


features = {}


def make_hook(index):

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


def run_rsu(feature_dict):

    cache = [
        None
    ] * len(model.model)

    cache[10] = feature_dict[10]
    cache[14] = feature_dict[14]
    cache[16] = feature_dict[16]

    x = feature_dict[16]

    drivable_raw = None

    with torch.no_grad():

        for i in range(17, 34):

            block = model.model[i]

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
                        x if j == -1 else cache[j]
                        for j in block.from_
                    ]

            x = block(x)

            if i == 33:
                drivable_raw = x

            cache[i] = x

    return torch.sigmoid(
        drivable_raw
    )


def make_fp16_features(source):

    result = {}

    for index in FEATURE_LAYERS:

        transmitted = (
            source[index].half()
        )

        result[index] = (
            transmitted.float()
        )

    return result


def make_int8_features(source):

    result = {}

    for index in FEATURE_LAYERS:

        feature = source[index]

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

        quantized = torch.clamp(
            torch.round(
                feature / scale
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
stats = {
    "FP32": {
        "intersection": 0,
        "union": 0,
        "correct": 0,
        "pixels": 0
    },
    "FP16": {
        "intersection": 0,
        "union": 0,
        "correct": 0,
        "pixels": 0
    },
    "INT8": {
        "intersection": 0,
        "union": 0,
        "correct": 0,
        "pixels": 0
    }
}


def update_stats(
    name,
    prediction,
    gt,
    valid
):

    pred = prediction[
        valid
    ].astype(bool)

    truth = gt[
        valid
    ].astype(bool)

    intersection = np.logical_and(
        pred,
        truth
    ).sum()

    union = np.logical_or(
        pred,
        truth
    ).sum()

    correct = (
        pred == truth
    ).sum()

    stats[name]["intersection"] += int(
        intersection
    )

    stats[name]["union"] += int(
        union
    )

    stats[name]["correct"] += int(
        correct
    )

    stats[name]["pixels"] += int(
        truth.size
    )


image_paths = sorted(
    IMAGE_DIR.glob("*.jpg")
)[:NUM_IMAGES]

print(
    "Images:",
    len(image_paths)
)


for number, image_path in enumerate(
    image_paths,
    start=1
):

    image_id = image_path.stem

    mask_path = (
        MASK_DIR
        / f"{image_id}_drivable_id.png"
    )

    if not mask_path.exists():

        print(
            "Mask missing:",
            mask_path
        )

        continue


    features.clear()


    image, info = load_image(
        image_path
    )

    gt, valid = load_gt_mask(
        mask_path,
        info
    )


    # ==========================================
    # FP32
    # ==========================================
    with torch.no_grad():

        normal_output = model(
            image
        )


    fp32_drivable = (
        normal_output[1]
    )


    original_features = {

        index:
        features[index].clone()

        for index
        in FEATURE_LAYERS
    }


    # ==========================================
    # FP16
    # ==========================================
    fp16_features = (
        make_fp16_features(
            original_features
        )
    )

    fp16_drivable = run_rsu(
        fp16_features
    )


    # ==========================================
    # INT8
    # ==========================================
    int8_features = (
        make_int8_features(
            original_features
        )
    )

    int8_drivable = run_rsu(
        int8_features
    )


    # ==========================================
    # 2クラス出力
    # 0 = non-drivable
    # 1 = drivable
    # ==========================================
    fp32_prediction = (
        torch.argmax(
            fp32_drivable,
            dim=1
        )[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    fp16_prediction = (
        torch.argmax(
            fp16_drivable,
            dim=1
        )[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    int8_prediction = (
        torch.argmax(
            int8_drivable,
            dim=1
        )[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    update_stats(
        "FP32",
        fp32_prediction,
        gt,
        valid
    )

    update_stats(
        "FP16",
        fp16_prediction,
        gt,
        valid
    )

    update_stats(
        "INT8",
        int8_prediction,
        gt,
        valid
    )


    if (
        number == 1
        or number % 10 == 0
    ):

        print(
            f"Processed "
            f"{number}/{NUM_IMAGES}"
        )


for hook in hooks:
    hook.remove()


print()
print(
    "======================================"
)

print(
    "DRIVABLE AREA EVALUATION"
)

print(
    "======================================"
)


for name in [
    "FP32",
    "FP16",
    "INT8"
]:

    intersection = (
        stats[name]["intersection"]
    )

    union = (
        stats[name]["union"]
    )

    correct = (
        stats[name]["correct"]
    )

    pixels = (
        stats[name]["pixels"]
    )


    if union > 0:

        iou = (
            intersection / union
        )

    else:

        iou = 0.0


    if pixels > 0:

        accuracy = (
            correct / pixels
        )

    else:

        accuracy = 0.0


    print()
    print(name)

    print(
        "Drivable IoU:",
        iou
    )

    print(
        "Pixel Accuracy:",
        accuracy
    )


total_elements = sum(
    original_features[index]
    .numel()

    for index
    in FEATURE_LAYERS
)


print()
print(
    "======================================"
)

print(
    "TRANSMISSION SIZE"
)

print(
    "======================================"
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
