from pathlib import Path

import numpy as np
from PIL import Image


# ==================================================
# 確認する画像
# ==================================================
IMAGE_ID = "b1c66a42-6f7d68ca"

IMAGE_PATH = Path(
    f"/home/tada/Application/datasets/"
    f"bdd100k/100k/val/{IMAGE_ID}.jpg"
)

MASK_PATH = Path(
    f"/home/tada/Application/datasets/"
    f"bdd100k/labels/val/"
    f"{IMAGE_ID}_drivable_id.png"
)

OUTPUT_DIR = Path(
    "/home/tada/Application/YOLOP/"
    "drivable_mask_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 読み込み
# ==================================================
image = Image.open(
    IMAGE_PATH
).convert("RGB")

mask = Image.open(
    MASK_PATH
)

image_np = np.array(
    image
)

mask_np = np.array(
    mask
)


print(
    "Image size:",
    image.size
)

print(
    "Mask shape:",
    mask_np.shape
)

print(
    "Unique mask values:",
    np.unique(mask_np)
)


# ==================================================
# 元画像保存
# ==================================================
image.save(
    OUTPUT_DIR / "original.jpg"
)


# ==================================================
# 各値だけを色付きで表示
#
# 0 = 赤
# 1 = 緑
# 2 = 青
# ==================================================
colors = {
    0: np.array([255, 0, 0]),
    1: np.array([0, 255, 0]),
    2: np.array([0, 100, 255]),
}


for value in [
    0,
    1,
    2
]:

    selected = (
        mask_np == value
    )

    overlay = (
        image_np.copy()
    )

    overlay[
        selected
    ] = colors[value]


    blended = (
        image_np.astype(
            np.float32
        )
        * 0.55
        +
        overlay.astype(
            np.float32
        )
        * 0.45
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )


    output = Image.fromarray(
        blended
    )

    output_path = (
        OUTPUT_DIR
        / f"mask_value_{value}.jpg"
    )

    output.save(
        output_path
    )


    pixel_count = int(
        selected.sum()
    )

    percentage = (
        pixel_count
        / mask_np.size
        * 100
    )


    print(
        f"value {value}: "
        f"{pixel_count} pixels "
        f"({percentage:.2f}%)"
    )


# ==================================================
# 3値を一度に色分けした画像
# ==================================================
combined = image_np.copy()

combined[
    mask_np == 0
] = (
    255,
    0,
    0
)

combined[
    mask_np == 1
] = (
    0,
    255,
    0
)

combined[
    mask_np == 2
] = (
    0,
    100,
    255
)


combined_blended = (
    image_np.astype(
        np.float32
    )
    * 0.45
    +
    combined.astype(
        np.float32
    )
    * 0.55
)

combined_blended = np.clip(
    combined_blended,
    0,
    255
).astype(
    np.uint8
)


Image.fromarray(
    combined_blended
).save(
    OUTPUT_DIR
    / "all_values.jpg"
)


print()
print(
    "Saved to:",
    OUTPUT_DIR
)

print()
print(
    "Color:"
)

print(
    "value 0 = RED"
)

print(
    "value 1 = GREEN"
)

print(
    "value 2 = BLUE"
)
