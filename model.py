import ailia

import numpy as np
import tempfile
import cv2
import os
import urllib.request

import sys
import time
from collections import OrderedDict

from PIL import Image

sys.path.append('util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clothing-detection/'

DATASETS_MODEL_PATH = OrderedDict([
    (
        'modanet',
        ['yolov3-modanet.opt.onnx', 'yolov3-modanet.opt.onnx.prototxt']
    ),
    ('df2', ['yolov3-df2.opt.onnx', 'yolov3-df2.opt.onnx.prototxt'])
])

DATASETS_CATEGORY = {
    'modanet': [
        "bag", "belt", "boots", "footwear", "outer", "dress", "sunglasses",
        "pants", "top", "shorts", "skirt", "headwear", "scarf/tie"
    ],
    'df2': [
        "short sleeve top", "long sleeve top", "short sleeve outwear",
        "long sleeve outwear", "vest", "sling", "shorts", "trousers", "skirt",
        "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
    ]
}

model_path = "yolov3.opt.onnx.prototxt"
weight_path = "yolov3.opt.onnx"
img_path = "pessoas.png"

print("downloading ...")

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov3/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov3/"+weight_path,weight_path)

env_id = ailia.get_gpu_environment_id()
categories = 80
detector = ailia.Detector(model_path, weight_path, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_U_FP32, algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3, env_id=env_id)

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED )
output_img = img.copy()
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

print( "img.shape=" + str(img.shape) )

w, h = img.shape[1], img.shape[0]

threshold = 0.4
iou = 0.45
detector.compute(img, threshold, iou)

coco_category=["person"]

count = detector.get_object_count()

print("object_count=" + str(count))

n_imgs = 0

for idx in range(count):
    obj = detector.get_object(idx)

    # Verificar se o objeto tem uma área detectada
    if obj.w > 0 and obj.h > 0:
        # Criar uma nova imagem contendo apenas o retângulo detectado
        roi = output_img[int(h*obj.y):int(h*(obj.y+obj.h)), int(w*obj.x):int(w*(obj.x+obj.w))]

        # Verificar se a região contém pixels antes de salvar
        if roi.size != 0:
            # Salvar a nova imagem com o nome baseado no índice atual
            n_imgs += 1
            cv2.imwrite("rectangle_" + str(idx) + ".jpg", roi)
            
THRESHOLD = 0.39
IOU = 0.4
DETECTION_WIDTH = 416

print(n_imgs)

def letterbox_image(image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

def preprocess(img, resize):
    image = Image.fromarray(img)
    boxed_image = letterbox_image(image, (resize, resize))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = np.transpose(image_data, [0, 3, 1, 2])
    return image_data


def post_processing(img_shape, all_boxes, all_scores, indices):
    indices = indices.astype(int)

    bboxes = []
    for idx_ in indices[0]:
        cls_ind = idx_[1]
        score = all_scores[tuple(idx_)]

        idx_1 = (idx_[0], idx_[2])
        box = all_boxes[idx_1]
        y, x, y2, x2 = box
        w = (x2 - x) / img_shape[1]
        h = (y2 - y) / img_shape[0]
        x /= img_shape[1]
        y /= img_shape[0]

        r = ailia.DetectorObject(
            category=cls_ind, prob=score,
            x=x, y=y, w=w, h=h,
        )
        bboxes.append(r)

    return bboxes

# ======================
# Main functions
# ======================

def detect_objects(img, detector):
    img_shape = img.shape[:2]

    # initial preprocesses
    img = preprocess(img, resize=DETECTION_WIDTH)

    # feedforward
    all_boxes, all_scores, indices = detector.predict({
        'input_1': img,
        'image_shape': np.array([img_shape], np.float32),
        'layer.score_threshold': np.array([THRESHOLD], np.float32),
        'iou_threshold': np.array([IOU], np.float32),
    })

    # post processes
    detect_object = post_processing(img_shape, all_boxes, all_scores, indices)

    return detect_object


def recognize_from_image(filename, detector):
    # prepare input data
    img = load_image(filename)
    logger.debug(f'input image shape: {img.shape}')

    x = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            detect_object = detect_objects(x, detector)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        detect_object = detect_objects(x, detector)

    # plot result
    res_img = plot_results(detect_object, img, category)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res_img)

def main():
        # model files check and download
        check_and_download_models(weight_path, model_path, REMOTE_PATH)

        # initialize
        detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
        print(args.env_id)
        id_image_shape = detector.find_blob_index_by_name("image_shape")
        detector.set_input_shape(
            (1, 3, DETECTION_WIDTH, DETECTION_WIDTH)
        )
        detector.set_input_blob_shape((1, 2), id_image_shape)

        # image mode
        # input image loop
        for image_path in args.input:
            # prepare input data
            logger.info(image_path)
            recognize_from_image(image_path, detector)

        logger.info('Script finished successfully.')

for i in range(0,n_imgs):

    IMAGE_PATH = 'rectangle_{}.jpg'.format(i)
    print(IMAGE_PATH)

    weight_path, model_path = DATASETS_MODEL_PATH['df2']
    category = DATASETS_CATEGORY['df2']

    
    SAVE_IMAGE_PATH = 'output_{}_df2.png'.format(i)

    parser = get_base_parser(
        'Clothing detection model', IMAGE_PATH, SAVE_IMAGE_PATH
    )
    args = update_parser(parser)

    if __name__ == '__main__':
        main()

    weight_path, model_path = DATASETS_MODEL_PATH['modanet']
    category = DATASETS_CATEGORY['modanet']

    SAVE_IMAGE_PATH = 'output_{}_modanet.png'.format(i)

    parser = get_base_parser(
        'Clothing detection model', IMAGE_PATH, SAVE_IMAGE_PATH
    )
    args = update_parser(parser)

    if __name__ == '__main__':
        main()