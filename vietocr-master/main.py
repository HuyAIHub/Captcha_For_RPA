import argparse
from PIL import Image
import base64,io
import requests
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
import time
import yaml
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import HTMLResponse
# with open('/home/huydq/PycharmProjects/RPA_Captcha_OCR/vietocr-master/vgg-transformer.yml', encoding='utf-8') as f:
#     config = yaml.safe_load(f)
# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = '/home/huydq/PycharmProjects/RPA_Captcha_OCR/transformerocr.pth'
# config['device'] = 'cpu'
# config['vocab'] = '3cghad6bfrp72ewxm5k84yn'
# print('config',config)
# print('config1',config1)

with open('config.yml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

detector = Predictor(config)

app = FastAPI()

@app.get("/") # giống flask, khai báo phương thức get và url
async def root(): # do dùng ASGI nên ở đây thêm async, nếu bên thứ 3 không hỗ trợ thì bỏ async đi
    return {"message": "Hello World"}

@app.post("/upload",response_class=HTMLResponse)
def upload(filedata: str = Form(...)):
    image_as_bytes = str.encode(filedata)  # convert string to bytes
    img_recovered = base64.b64decode(image_as_bytes)  # decode base64string
    image = Image.open(io.BytesIO(img_recovered))
    image_np = np.array(image)
    try:
        txt = main(image_np)
    except Exception:
        return {"message": "There was an error uploading the file"}
    return HTMLResponse(content=txt,status_code = 200)

def Preprocessing(Image):
  # Image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
  #convert no background transparent to white background transparent
  mask = Image[:,:,3] == 0
  Image[mask] = [255,255,255,255]
  #removing gird
  img = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
  kernel = np.ones((3, 3), np.uint8)
  # dilation increases the object area
  img_dilation = cv2.dilate(img, kernel, iterations=1)
  # erosion trying to keep foreground in white
  img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
  return img_erosion
# cv2.imshow('oke',img_recovered)
def main(image):
    t1 = time.time()
    img = Preprocessing(image)
    img = Image.fromarray(img)
    print('img2:',img)
    text = detector.predict(img)
    t2 = time.time()
    print('text_predict:',text)
    print('time:',t2-t1)
    return text

if __name__ == '__main__':
    main() 
