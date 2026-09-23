#%% IMPORTS AND REQUIRED SET-UP

from pathlib import Path
from PIL import Image

import numpy as np
import onnxruntime


IMG_DIR_PATH = r"/mnt/c/Users/Akshay/Desktop/Coursework/Semester 3/MTP/github/mobilenet/sample_data" #set the path to the directory with the samples
IMAGENET_CLASS_PATH = r"/mnt/c/Users/Akshay/Desktop/Coursework/Semester 3/MTP/github/mobilenet/imagenet_classes.txt" #file that lists imagenet classes to index


def preprocess_image(image_path: Path,
                     height: int,
                     width: int,
                     num_channels: int = 3):

    # for inferencing on the PI, re-writing the transforms using numpy and PIL
    # ref: https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html

    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS) # https://stackoverflow.com/questions/1854146/what-is-the-idea-behind-scaling-an-image-using-lanczos
    
    image_np = np.asarray(image).astype(np.float32)
    image_np = image_np.transpose([2,0,1]) # (Channel, Height, Width) is how torch expects it

    # transform the RGB values using the ImageNet mean and std
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224

    for n in range(num_channels):
        image_np[n, :,:] = (image_np[n, :,:] / 255 - mean[n]) / std[n]

    image_np = np.expand_dims(image_np, 0) # [1, C, H, W]
    return image_np

def softmax(x):
      """Compute softmax values for each sets of scores in x."""
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()


def process_image(ort_session, image_path, imagenet_categories):

    image_np = preprocess_image(image_path, height=224, width=224)

    # although image_np is 4 DIM, make sure to wrap it in a list
    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), [image_np])}
    
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
    onnxruntime_outputs = onnxruntime_outputs.flatten()

    output_probabilities = softmax(onnxruntime_outputs)

    top5_categories = np.argsort(-output_probabilities)[:5]

    print(image_path)
    for catid in top5_categories:
        print(imagenet_categories[catid], output_probabilities[catid])



#%% MAIN

if __name__ == "__main__":


    folder = Path(IMG_DIR_PATH)
    jpeg_files = list(folder.glob("*.JPEG")) + list(folder.glob("*.jpeg"))

    if not jpeg_files:
        raise FileNotFoundError(f"Input Images not Found at given path: {folder}") 

    with open(IMAGENET_CLASS_PATH, "r") as f:
        imagenet_categories = [s.strip() for s in f.readlines()]

    ort_session = onnxruntime.InferenceSession(
                            "./mobilenet_v2.onnx", providers=["CPUExecutionProvider"]
                        )

    for image_path in jpeg_files:
        process_image(ort_session, image_path, imagenet_categories)
        print('\n')
    