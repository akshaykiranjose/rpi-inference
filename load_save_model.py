#%%
import torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import functional as Func
torch.manual_seed(1337)

data_transforms = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()

#%% CREATE THE MODEL AND RUN INFERENCE ON A SINGLE IMAGE
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

img_tensor = Func.pil_to_tensor(Image.open('../data/imagenet1k/ILSVRC2012_val_00000057.JPEG'))
img_tensor = data_transforms(img_tensor)
img_transformed = data_transforms(img_tensor)

img_input = img_transformed.unsqueeze(0)
img_output = model(img_input)


# %% CONVERT THE MODEL TO ONNX AND REPEAT THE INFERENCE

example_inputs = (img_input, )
model.eval() # be-sure to use this to avoid a UserWarning

onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
onnx_program.save("mobilenet_v2.onnx")

#%% START AN ONNX-RUNTIME
import onnxruntime

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
print(f"Input length: {len(onnx_inputs)}")
print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    "./mobilenet_v2.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

#%%
torch_outputs = model(*example_inputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")

# %%
