import uvicorn
from fastapi import FastAPI,File
import tensorflow as tf
import torch
import base64
from PIL import Image
from itertools import product
from PIL import ImageFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import asyncio
 
# convert into JSON:

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_yolov5():
    model = torch.hub.load('./yolov5', 'custom', path='brand_weights.pt', source='local' ,force_reload=True) 
    model.conf = 0.5
    return model


app = FastAPI()


model = get_yolov5()


origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_image_from_bytes(binary_image):
    input_image = Image.open(BytesIO(binary_image))
    return input_image


@app.get('/')
def home():
    return 'hello'

@asyncio.coroutine
@app.post('/predict')
def predict(file: bytes = File(...)):
    # data=data.dict()
    # image = base64.b64decode(data['image'])
    image = get_image_from_bytes(file)
    results = model(image)
    results.render()
    # print(results.pandas().xyxy[0]['name'])
    img_str = ""
    for img in results.ims:
        pil_img=Image.fromarray(img)
        # pil_img.save("temp.png")
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
    return {
        "image" : img_str
    }
    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)  