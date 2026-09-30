from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img

from lib.core.general import (
    non_max_suppression,
    box_iou,
    ap_per_class
)


NUM_IMAGES = 100

IMAGE_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/100k/val"
)

MASK_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/labels/val"
)

LABEL_DIR = Path(
    "/home/tada/Application/datasets/"
    "bdd100k_det_yolo/val/labels"
)

IMPORTANCE_FILE = (
    "channel_importance.pt"
)

FEATURE_LAYERS = [
    10,
    14,
    16
]


# ==================================================
# 今回比較する条件
# ==================================================
EXPERIMENTS = [
    {
        "name": "FP32_100",
        "keep_ratio": 1.00,
        "use_int8": False
    },
    {
        "name": "INT8_100",
        "keep_ratio": 1.00,
        "use_int8": True
    },
    {
        "name": "INT8_95",
        "keep_ratio": 0.95,
        "use_int8": True
    },
    {
        "name": "INT8_90",
        "keep_ratio": 0.90,
        "use_int8": True
    },
    {
        "name": "INT8_85",
        "keep_ratio": 0.85,
        "use_int8": True
    }
]


# ==================================================
# Device
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
# Channel importance
# ==================================================
importance_data = torch.load(
    IMPORTANCE_FILE,
    map_location="cpu"
)

rankings = {
    index:
        importance_data[index]["ranking"]

    for index
    in FEATURE_LAYERS
}

print(
    "Channel importance loaded."
)


# ==================================================
# Normalize
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
# Image preprocessing
# ==================================================
def load_image(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: "
            f"{image_path}"
        )


    h0, w0 = (
        image.shape[:2]
    )


    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    processed, _, _ = (
        letterbox_for_img(
            image_rgb,
            new_shape=640,
            auto=True
        )
    )


    input_h, input_w = (
        processed.shape[:2]
    )


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


    pad_w = (
        input_w
        - resized_w
    )

    pad_h = (
        input_h
        - resized_h
    )


    left = int(
        round(
            pad_w / 2 - 0.1
        )
    )

    top = int(
        round(
            pad_h / 2 - 0.1
        )
    )


    processed = (
        processed.transpose(
            2,
            0,
            1
        )
    )

    processed = (
        np.ascontiguousarray(
            processed
        )
    )


    tensor = torch.from_numpy(
        processed
    ).float()

    tensor /= 255.0

    tensor = normalize(
        tensor
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )


    info = {
        "orig_h": h0,
        "orig_w": w0,
        "input_h": input_h,
        "input_w": input_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "top": top,
        "left": left,
        "scale": r
    }


    return (
        tensor,
        info
    )


# ==================================================
# Drivable GT
# ==================================================
def load_drivable_gt(
    mask_path,
    info
):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        raise RuntimeError(
            f"Could not read mask: "
            f"{mask_path}"
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
        top:
        top + info["resized_h"],
        left:
        left + info["resized_w"]
    ] = gt_resized


    valid = np.zeros_like(
        gt_input,
        dtype=bool
    )


    valid[
        top:
        top + info["resized_h"],
        left:
        left + info["resized_w"]
    ] = True


    return (
        gt_input,
        valid
    )


# ==================================================
# Vehicle Detection GT
# ==================================================
VEHICLE_CLASSES = {
    2,
    3,
    4,
    9
}


def load_vehicle_gt(
    label_path,
    info
):

    boxes = []


    if not label_path.exists():

        return torch.zeros(
            (0, 5),
            dtype=torch.float32,
            device=device
        )


    with open(
        label_path,
        "r"
    ) as f:

        lines = f.readlines()


    for line in lines:

        parts = (
            line.strip()
            .split()
        )


        if len(parts) != 5:
            continue


        cls = int(
            float(
                parts[0]
            )
        )


        if cls not in VEHICLE_CLASSES:
            continue


        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])


        x1 = (
            cx - bw / 2
        ) * info["orig_w"]

        y1 = (
            cy - bh / 2
        ) * info["orig_h"]

        x2 = (
            cx + bw / 2
        ) * info["orig_w"]

        y2 = (
            cy + bh / 2
        ) * info["orig_h"]


        x1 = (
            x1
            * info["scale"]
            + info["left"]
        )

        y1 = (
            y1
            * info["scale"]
            + info["top"]
        )

        x2 = (
            x2
            * info["scale"]
            + info["left"]
        )

        y2 = (
            y2
            * info["scale"]
            + info["top"]
        )


        boxes.append(
            [
                0,
                x1,
                y1,
                x2,
                y2
            ]
        )


    if len(boxes) == 0:

        return torch.zeros(
            (0, 5),
            dtype=torch.float32,
            device=device
        )


    return torch.tensor(
        boxes,
        dtype=torch.float32,
        device=device
    )


# ==================================================
# F10 / F14 / F16取得
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
            output
            .detach()
            .clone()
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
# チャネル削減
# ==================================================
def reduce_channels(
    source,
    keep_ratio
):

    result = {}


    for index in FEATURE_LAYERS:

        feature = (
            source[index]
        )

        total_channels = (
            feature.shape[1]
        )

        keep_count = int(
            total_channels
            * keep_ratio
        )

        keep_indices = (
            rankings[index][
                :keep_count
            ]
            .to(device)
        )

        reduced = (
            torch.zeros_like(
                feature
            )
        )

        reduced[
            :,
            keep_indices,
            :,
            :
        ] = feature[
            :,
            keep_indices,
            :,
            :
        ]

        result[index] = (
            reduced
        )


    return result


# ==================================================
# INT8量子化
# ==================================================
def quantize_int8(
    source
):

    result = {}


    for index in FEATURE_LAYERS:

        feature = (
            source[index]
        )


        max_value = (
            feature
            .abs()
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
            quantized
            .float()
            * scale
        )


    return result
# ==================================================
# RSU側処理
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

    x = (
        feature_dict[16]
    )

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
# Detection IoU thresholds
# ==================================================
iouv = torch.linspace(
    0.5,
    0.95,
    10
).to(device)

niou = iouv.numel()


# ==================================================
# Detection matching
# ==================================================
def match_predictions(
    pred,
    gt
):

    correct = torch.zeros(
        (
            pred.shape[0],
            niou
        ),
        dtype=torch.bool,
        device=device
    )


    if (
        len(pred) == 0
        or len(gt) == 0
    ):

        return correct


    target_classes = (
        gt[:, 0]
    )

    target_boxes = (
        gt[:, 1:5]
    )

    detected_targets = set()


    for cls in torch.unique(
        target_classes
    ):

        target_indices = (
            target_classes == cls
        ).nonzero(
            as_tuple=False
        ).view(-1)

        prediction_indices = (
            pred[:, 5] == cls
        ).nonzero(
            as_tuple=False
        ).view(-1)

        if prediction_indices.numel() == 0:

            continue


        ious, best_target = box_iou(
            pred[
                prediction_indices,
                :4
            ],
            target_boxes[
                target_indices
            ]
        ).max(
            dim=1
        )


        order = torch.argsort(
            pred[
                prediction_indices,
                4
            ],
            descending=True
        )


        for order_index in order:

            pred_index = (
                prediction_indices[
                    order_index
                ]
            )

            iou_value = (
                ious[
                    order_index
                ]
            )

            local_target_index = (
                best_target[
                    order_index
                ]
            )

            target_index = (
                target_indices[
                    local_target_index
                ]
            )


            if (
                iou_value
                < iouv[0]
            ):

                continue


            target_number = int(
                target_index.item()
            )


            if (
                target_number
                in detected_targets
            ):

                continue


            detected_targets.add(
                target_number
            )


            correct[
                pred_index
            ] = (
                iou_value
                >= iouv
            )


    return correct


# ==================================================
# 結果保存
# ==================================================
results_store = {}


for experiment in EXPERIMENTS:

    name = experiment[
        "name"
    ]

    results_store[name] = {
        "det_stats": [],
        "intersection": 0,
        "union": 0,
        "correct_pixels": 0,
        "total_pixels": 0,
        "targets": 0
    }


# ==================================================
# Drivable統計
# ==================================================
def update_drivable_stats(
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


    results_store[
        name
    ]["intersection"] += int(
        intersection
    )

    results_store[
        name
    ]["union"] += int(
        union
    )

    results_store[
        name
    ]["correct_pixels"] += int(
        correct
    )

    results_store[
        name
    ]["total_pixels"] += int(
        truth.size
    )


# ==================================================
# Detection統計
# ==================================================
def update_detection_stats(
    name,
    raw_detection,
    gt
):

    detection_tensor = (
        get_detection_tensor(
            raw_detection
        )
    )


    outputs = non_max_suppression(
        detection_tensor,
        conf_thres=(
            cfg.TEST.NMS_CONF_THRESHOLD
        ),
        iou_thres=(
            cfg.TEST.NMS_IOU_THRESHOLD
        )
    )


    pred = outputs[0]


    target_classes = (
        gt[:, 0]
        if len(gt)
        else torch.zeros(
            0,
            device=device
        )
    )


    results_store[
        name
    ]["targets"] += len(
        target_classes
    )


    if len(pred) == 0:

        correct = torch.zeros(
            (
                0,
                niou
            ),
            dtype=torch.bool
        )

        results_store[
            name
        ]["det_stats"].append(
            (
                correct,
                torch.zeros(0),
                torch.zeros(0),
                target_classes
                .detach()
                .cpu()
            )
        )

        return


    correct = match_predictions(
        pred,
        gt
    )


    results_store[
        name
    ]["det_stats"].append(
        (
            correct
            .detach()
            .cpu(),

            pred[:, 4]
            .detach()
            .cpu(),

            pred[:, 5]
            .detach()
            .cpu(),

            target_classes
            .detach()
            .cpu()
        )
    )


# ==================================================
# 評価画像
# ==================================================
image_paths = sorted(
    IMAGE_DIR.glob(
        "*.jpg"
    )
)[:NUM_IMAGES]


print(
    "Images:",
    len(image_paths)
)


# ==================================================
# 実験
# ==================================================
for number, image_path in enumerate(
    image_paths,
    start=1
):

    image_id = (
        image_path.stem
    )


    mask_path = (
        MASK_DIR
        / f"{image_id}_drivable_id.png"
    )

    label_path = (
        LABEL_DIR
        / f"{image_id}.txt"
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


    gt_drivable, valid = (
        load_drivable_gt(
            mask_path,
            info
        )
    )


    gt_detection = (
        load_vehicle_gt(
            label_path,
            info
        )
    )


    # ==========================================
    # 元の特徴量取得
    # ==========================================
    with torch.no_grad():

        _ = model(
            image
        )


    original_features = {

        index:
        features[index]
        .clone()

        for index
        in FEATURE_LAYERS
    }


    # ==========================================
    # 各条件
    # ==========================================
    for experiment in EXPERIMENTS:

        name = (
            experiment[
                "name"
            ]
        )

        keep_ratio = (
            experiment[
                "keep_ratio"
            ]
        )

        use_int8 = (
            experiment[
                "use_int8"
            ]
        )


        current_features = (
            reduce_channels(
                original_features,
                keep_ratio
            )
        )


        if use_int8:

            current_features = (
                quantize_int8(
                    current_features
                )
            )


        detection, drivable = (
            run_rsu(
                current_features
            )
        )


        drivable_prediction = (
            torch.argmax(
                drivable,
                dim=1
            )[0]
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )


        update_drivable_stats(
            name,
            drivable_prediction,
            gt_drivable,
            valid
        )


        update_detection_stats(
            name,
            detection,
            gt_detection
        )


    if (
        number == 1
        or number % 10 == 0
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
# Detection結果
# ==================================================
def calculate_detection_result(
    name
):

    stats_list = (
        results_store[
            name
        ]["det_stats"]
    )


    correct = np.concatenate(
        [
            item[0].numpy()
            for item
            in stats_list
        ],
        axis=0
    )


    confidence = np.concatenate(
        [
            item[1].numpy()
            for item
            in stats_list
        ],
        axis=0
    )


    pred_class = np.concatenate(
        [
            item[2].numpy()
            for item
            in stats_list
        ],
        axis=0
    )


    target_class = np.concatenate(
        [
            item[3].numpy()
            for item
            in stats_list
        ],
        axis=0
    )


    if len(confidence) == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0
        )


    p, r, ap, f1, ap_class = (
        ap_per_class(
            correct,
            confidence,
            pred_class,
            target_class,
            plot=False
        )
    )


    precision = float(
        np.mean(p)
    )

    recall = float(
        np.mean(r)
    )


    if ap.size > 0:

        ap50 = float(
            np.mean(
                ap[:, 0]
            )
        )

        map5095 = float(
            np.mean(
                ap
            )
        )

    else:

        ap50 = 0.0
        map5095 = 0.0


    return (
        precision,
        recall,
        ap50,
        map5095
    )


# ==================================================
# 理論通信量
# ==================================================
def theoretical_size_mib(
    keep_ratio,
    use_int8
):

    total_bytes = 0


    for index in FEATURE_LAYERS:

        feature = (
            original_features[
                index
            ]
        )

        batch = (
            feature.shape[0]
        )

        channels = (
            feature.shape[1]
        )

        height = (
            feature.shape[2]
        )

        width = (
            feature.shape[3]
        )


        keep_channels = int(
            channels
            * keep_ratio
        )


        elements = (
            batch
            * keep_channels
            * height
            * width
        )


        if use_int8:

            bytes_per_value = 1

        else:

            bytes_per_value = 4


        total_bytes += (
            elements
            * bytes_per_value
        )


    return (
        total_bytes
        / 1024
        / 1024
    )


# ==================================================
# 最終表示
# ==================================================
print()
print(
    "======================================"
)

print(
    "INT8 + CHANNEL REDUCTION EVALUATION"
)

print(
    "======================================"
)


for experiment in EXPERIMENTS:

    name = (
        experiment[
            "name"
        ]
    )

    keep_ratio = (
        experiment[
            "keep_ratio"
        ]
    )

    use_int8 = (
        experiment[
            "use_int8"
        ]
    )


    (
        precision,
        recall,
        ap50,
        map5095

    ) = calculate_detection_result(
        name
    )


    intersection = (
        results_store[
            name
        ]["intersection"]
    )

    union = (
        results_store[
            name
        ]["union"]
    )

    correct_pixels = (
        results_store[
            name
        ]["correct_pixels"]
    )

    total_pixels = (
        results_store[
            name
        ]["total_pixels"]
    )


    if union > 0:

        drivable_iou = (
            intersection
            / union
        )

    else:

        drivable_iou = 0.0


    if total_pixels > 0:

        pixel_accuracy = (
            correct_pixels
            / total_pixels
        )

    else:

        pixel_accuracy = 0.0


    size_mib = (
        theoretical_size_mib(
            keep_ratio,
            use_int8
        )
    )


    print()
    print(name)

    print(
        "Transmission size:",
        size_mib,
        "MiB"
    )

    print(
        "Targets:",
        results_store[
            name
        ]["targets"]
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "Detection AP50:",
        ap50
    )

    print(
        "Detection mAP50-95:",
        map5095
    )

    print(
        "Drivable IoU:",
        drivable_iou
    )

    print(
        "Drivable Pixel Accuracy:",
        pixel_accuracy
    )
