from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img


NUM_IMAGES = 500

IMAGE_DIR = Path(
    "/home/tada/Application/datasets/bdd100k/100k/train"
)

OUTPUT_FILE = (
    "channel_importance.pt"
)

FEATURE_LAYERS = [
    10,
    14,
    16
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

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

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    return tensor


features = {}


def make_hook(index):

    def hook(
        module,
        inputs,
        output
    ):

        features[index] = (
            output.detach()
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


importance_sum = {
    10: None,
    14: None,
    16: None
}


image_paths = sorted(
    IMAGE_DIR.glob("*.jpg")
)[:NUM_IMAGES]


print(
    "Calibration images:",
    len(image_paths)
)


with torch.no_grad():

    for number, image_path in enumerate(
        image_paths,
        start=1
    ):

        features.clear()

        image = load_image(
            image_path
        )

        _ = model(
            image
        )


        for index in FEATURE_LAYERS:

            feature = features[index]

            # 各チャネルについて
            # 空間方向の絶対値平均を計算
            channel_score = (
                feature
                .abs()
                .mean(
                    dim=(0, 2, 3)
                )
                .cpu()
            )


            if importance_sum[index] is None:

                importance_sum[index] = (
                    channel_score.clone()
                )

            else:

                importance_sum[index] += (
                    channel_score
                )


        if (
            number == 1
            or number % 50 == 0
        ):

            print(
                f"Processed "
                f"{number}/{NUM_IMAGES}"
            )


for hook in hooks:
    hook.remove()


results = {}


print()
print(
    "======================================"
)

print(
    "CHANNEL IMPORTANCE"
)

print(
    "======================================"
)


for index in FEATURE_LAYERS:

    average_importance = (
        importance_sum[index]
        / len(image_paths)
    )


    ranking = torch.argsort(
        average_importance,
        descending=True
    )


    channels = len(
        average_importance
    )


    print()
    print(
        f"F{index}"
    )

    print(
        "Channels:",
        channels
    )

    print(
        "Most important 10 channels:",
        ranking[:10].tolist()
    )


    results[index] = {
        "importance":
            average_importance,

        "ranking":
            ranking
    }


torch.save(
    results,
    OUTPUT_FILE
)


print()
print(
    "Saved:",
    OUTPUT_FILE
)


print()
print(
    "Keep-channel plan"
)

for ratio in [
    0.75,
    0.50,
    0.25
]:

    print()
    print(
        f"{int(ratio * 100)}% keep"
    )

    for index in FEATURE_LAYERS:

        total = len(
            results[index][
                "ranking"
            ]
        )

        keep = int(
            total * ratio
        )

        print(
            f"F{index}: "
            f"{keep}/{total} channels"
        )
