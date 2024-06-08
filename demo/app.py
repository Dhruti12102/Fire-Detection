import gradio as gr
import torch
from PIL import Image
import pathlib
import os

os.chdir("./demo")
temp = pathlib.PosixPath
pathlib.PosixPath  = pathlib.WindowsPath

model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt" , force_reload=True)


def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.LANCZOS)  # resize
    results = model(im)
    results.render()
    return Image.fromarray(results.ims[0])


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "Fire Detection System"
description = "YOLOv5 demo for fire detection. Upload an image or click an example image to use."
article = ""
examples = [['pan-fire.jpg'], ['fire-basket.jpg'],['smokyfire.jpg'],['sunset.jpg'],['fireincar.jpg'],['fire0.jpg'],['fire1.jpg'],['fire2.jpg'],['fire3.jpg'],['fire4.jpg'],['fire5.jpg'],['fire6.jpg'],['fire7.jpg']]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(
    debug=True)

pathlib.PosixPath = temp