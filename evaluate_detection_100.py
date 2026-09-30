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
    ap_per_class,
    xywh2xyxy
)


NUM_IMAGES = 10000

IMAGE_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/100k/val"
)

LABEL_DIR = Path(
    "/home/tada/Application/datasets/bdd100k_det_yolo/val/labels"
)

FEATURE_LAYERS = [10, 14, 16]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


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


def load_image(
    image_path
):

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
# YOLO 10クラスのGTから
# vehicleだけ取り出す
#
# 2 = car
# 3 = bus
# 4 = truck
# 9 = train
#
# 最終的には全部 class 0
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
            (
                0,
                5
            ),
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


        cx = float(
            parts[1]
        )

        cy = float(
            parts[2]
        )

        bw = float(
            parts[3]
        )

        bh = float(
            parts[4]
        )


        # 元画像pixel座標
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


        # YOLOP入力画像座標へ変換
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
            (
                0,
                5
            ),
            dtype=torch.float32,
            device=device
        )


    return torch.tensor(
        boxes,
        dtype=torch.float32,
        device=device
    )


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


def get_detection_tensor(
    output
):

    if isinstance(
        output,
        tuple
    ):

        return output[0]

    return output


def run_rsu_detection(
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


    with torch.no_grad():

        for i in range(
            17,
            25
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


            cache[i] = x


            if i == 24:

                detection = x


    return detection


def make_fp16_features(
    source
):

    result = {}


    for index in FEATURE_LAYERS:

        result[index] = (
            source[index]
            .half()
            .float()
        )


    return result


def make_int8_features(
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
                max_value
                / 127.0
            )


        quantized = torch.clamp(
            torch.round(
                feature
                / scale
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
# IoU thresholds
# AP@0.5 ～ AP@0.95
# ==================================================
iouv = torch.linspace(
    0.5,
    0.95,
    10
).to(device)

niou = iouv.numel()


# ==================================================
# 予測BBoxと正解BBoxを対応付ける
#
# pred:
# [x1, y1, x2, y2, confidence, class]
#
# gt:
# [class, x1, y1, x2, y2]
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


    target_classes = gt[:, 0]

    target_boxes = gt[:, 1:5]


    detected_targets = set()


    # 今回は全てvehicle class 0
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


        # 各predictionについて
        # 一番IoUが高いGTを探す
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


        # confidenceの高いpredictionから処理
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

            target_local_index = (
                best_target[
                    order_index
                ]
            )

            target_index = (
                target_indices[
                    target_local_index
                ]
            )


            # IoU 0.5未満なら
            # どのthresholdでも不正解
            if iou_value < iouv[0]:

                continue


            target_number = int(
                target_index.item()
            )


            # 同じGTを2回使わない
            if target_number in detected_targets:

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
# 評価結果保存
# ==================================================
all_stats = {
    "FP32": [],
    "FP16": [],
    "INT8": []
}


target_count = {
    "FP32": 0,
    "FP16": 0,
    "INT8": 0
}


# ==================================================
# 1画像分の評価
# ==================================================
def evaluate_one(
    name,
    raw_detection,
    gt
):

    detection_tensor = (
        get_detection_tensor(
            raw_detection
        )
    )


    # YOLOP標準NMS
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


    target_count[name] += len(
        target_classes
    )


    # predictionなし
    if len(pred) == 0:

        correct = torch.zeros(
            (
                0,
                niou
            ),
            dtype=torch.bool
        )


        all_stats[name].append(
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


    # predictionあり
    correct = match_predictions(
        pred,
        gt
    )


    all_stats[name].append(
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
# 100枚
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


for number, image_path in enumerate(
    image_paths,
    start=1
):

    image_id = (
        image_path.stem
    )


    label_path = (
        LABEL_DIR
        / f"{image_id}.txt"
    )


    features.clear()


    # ==========================================
    # 画像
    # ==========================================
    image, info = load_image(
        image_path
    )


    # ==========================================
    # vehicle GT
    # ==========================================
    gt = load_vehicle_gt(
        label_path,
        info
    )


    # ==========================================
    # FP32
    # ==========================================
    with torch.no_grad():

        normal_output = model(
            image
        )


    fp32_detection = (
        normal_output[0]
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


    fp16_detection = (
        run_rsu_detection(
            fp16_features
        )
    )


    # ==========================================
    # INT8
    # ==========================================
    int8_features = (
        make_int8_features(
            original_features
        )
    )


    int8_detection = (
        run_rsu_detection(
            int8_features
        )
    )


    # ==========================================
    # 評価
    # ==========================================
    evaluate_one(
        "FP32",
        fp32_detection,
        gt
    )


    evaluate_one(
        "FP16",
        fp16_detection,
        gt
    )


    evaluate_one(
        "INT8",
        int8_detection,
        gt
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
# AP計算
# ==================================================
def calculate_results(
    name
):

    stats_list = (
        all_stats[name]
    )


    if len(stats_list) == 0:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "ap50": 0.0,
            "map": 0.0
        }


    correct = np.concatenate(
        [
            item[0].numpy()
            for item in stats_list
        ],
        axis=0
    )


    confidence = np.concatenate(
        [
            item[1].numpy()
            for item in stats_list
        ],
        axis=0
    )


    pred_class = np.concatenate(
        [
            item[2].numpy()
            for item in stats_list
        ],
        axis=0
    )


    target_class = np.concatenate(
        [
            item[3].numpy()
            for item in stats_list
        ],
        axis=0
    )


    # predictionが1件もない場合
    if len(confidence) == 0:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "ap50": 0.0,
            "map": 0.0
        }


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
            np.mean(ap)
        )

    else:

        ap50 = 0.0
        map5095 = 0.0


    return {
        "precision": precision,
        "recall": recall,
        "ap50": ap50,
        "map": map5095
    }


results = {}


for name in [
    "FP32",
    "FP16",
    "INT8"
]:

    results[name] = (
        calculate_results(
            name
        )
    )


# ==================================================
# 結果表示
# ==================================================
print()
print(
    "======================================"
)

print(
    "VEHICLE DETECTION EVALUATION"
)

print(
    "======================================"
)


for name in [
    "FP32",
    "FP16",
    "INT8"
]:

    result = results[name]


    print()
    print(name)

    print(
        "Targets:",
        target_count[name]
    )

    print(
        "Precision:",
        result["precision"]
    )

    print(
        "Recall:",
        result["recall"]
    )

    print(
        "AP50:",
        result["ap50"]
    )

    print(
        "mAP50-95:",
        result["map"]
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
