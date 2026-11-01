from build_model import build_model, count_parameters

print("===== ROADGUARD AI - MODEL 2 =====")

print("\nLoading ImageNet-pretrained ResNet50 weights...")
model = build_model("pretrained")

total, trainable = count_parameters(model)

print("\nArchitecture: Faster R-CNN ResNet50-FPN")
print("Initialization: ImageNet-pretrained backbone")
print("Number of classes: 4")
print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")

print("\nModel 2 created successfully.")