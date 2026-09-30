from pathlib import Path


IMAGE_DIR = Path(
    "/home/tada/Application/datasets/"
    "bdd100k/100k/train"
)

NUM_IMAGES = 1000


images = sorted(
    IMAGE_DIR.glob("*.jpg")
)[:NUM_IMAGES]


sizes = [
    p.stat().st_size
    for p in images
]


total_bytes = sum(sizes)

average_bytes = (
    total_bytes / len(sizes)
)

min_bytes = min(sizes)
max_bytes = max(sizes)


print(
    "Images checked:",
    len(images)
)

print()

print(
    "Average JPEG size:"
)

print(
    f"{average_bytes / 1024:.2f} KiB"
)

print(
    f"{average_bytes / 1024 / 1024:.4f} MiB"
)

print()

print(
    "Minimum JPEG size:",
    f"{min_bytes / 1024:.2f} KiB"
)

print(
    "Maximum JPEG size:",
    f"{max_bytes / 1024:.2f} KiB"
)


# ==========================================
# Shared featureとの比較
# ==========================================
feature_fp32_mib = 7.421875
feature_fp16_mib = 3.7109375

jpeg_mib = (
    average_bytes
    / 1024
    / 1024
)


print()
print(
    "===== Comparison ====="
)

print(
    f"Average JPEG : {jpeg_mib:.4f} MiB"
)

print(
    f"Feature FP32 : {feature_fp32_mib:.4f} MiB"
)

print(
    f"Feature FP16 : {feature_fp16_mib:.4f} MiB"
)

print()

print(
    "FP32 feature / JPEG:",
    f"{feature_fp32_mib / jpeg_mib:.2f}x"
)

print(
    "FP16 feature / JPEG:",
    f"{feature_fp16_mib / jpeg_mib:.2f}x"
)
