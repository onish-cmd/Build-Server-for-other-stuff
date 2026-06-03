import torch
import torch.onnx

# 1. Import your model architecture definition here
# from model import MobileFaceNet 

def convert():
    # Load your model structure (ensure it matches the .pth weights)
    model = MobileFaceNet() 
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()

    # 2. Define input shape (1, 3, 128, 128)
    # The '1' is the batch size, which OpenCV expects.
    dummy_input = torch.randn(1, 3, 128, 128)

    # 3. Export
    # opset_version=11 is the most stable version for OpenCV's DNN module
    torch.onnx.export(
        model, 
        dummy_input, 
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model successfully converted to model.onnx")

if __name__ == "__main__":
    convert()
