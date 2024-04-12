MODEL_CODE_TO_NAME: dict[int, str] = {
    0: "DebugNet",
    1: "KyleNet",
    3: "SqueezeNet",
    18: "ResNet-18",
    50: "ResNet-50",
    152: "ResNet-152",
}

MODEL_NAME_TO_CODE: dict[str, int] = {
    name: code
    for code, name in MODEL_CODE_TO_NAME.items()
}