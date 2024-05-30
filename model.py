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


REMOTE_PATH_CLOTHING_DETECTION = 'https://storage.googleapis.com/ailia-models/clothing-detection/'
REMOTE_PATH_YOLO = "https://storage.googleapis.com/ailia-models/yolov3/"

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



def download_models(model_path, weight_path):
    print("Downloading models...")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(REMOTE_PATH_YOLO + model_path, model_path)
    if not os.path.exists(weight_path):
        urllib.request.urlretrieve(REMOTE_PATH_YOLO + weight_path, weight_path)

def initialize_yolo_detector(model_path, weight_path):

    download_models(model_path, weight_path)

    env_id = ailia.get_gpu_environment_id()
    categories = 80  
    return ailia.Detector(model_path, weight_path, categories,
                          format=ailia.NETWORK_IMAGE_FORMAT_RGB,
                          channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
                          range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
                          algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
                          env_id=env_id)

def detect_people(img_path):

    detector = initialize_yolo_detector(model_path, weight_path)

    threshold = 0.4
    iou = 0.45

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    output_img = img.copy()
    if img.shape[2] == 3 :
        img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
    elif img.shape[2] == 1 : 
        img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

    w, h = img.shape[1], img.shape[0]
    detector.compute(img, threshold, iou)
    count = detector.get_object_count()

    detected_people = []
    for idx in range(count):
        obj = detector.get_object(idx)

        if obj.w > 0 and obj.h > 0 and obj.category == 0:

            roi = output_img[int(h * obj.y):int(h * (obj.y + obj.h)), int(w * obj.x):int(w * (obj.x + obj.w))]

            if roi.size != 0:
                detected_people.append(roi)

    return detected_people

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

def detect_objects(img, detector):
        
        THRESHOLD = 0.39
        IOU = 0.4
        DETECTION_WIDTH = 416

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

## RECEBE ARGS E CATEGORY
def recognize_from_image(img, detector, args, category):
    # prepare input data
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

    res_img, results = plot_results(detect_object, img, category)

    return results

def detect_clothes(img, weight_path, model_path, args, category):
            
    DETECTION_WIDTH = 416

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH_CLOTHING_DETECTION)

    # initialize
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
    
    id_image_shape = detector.find_blob_index_by_name("image_shape")
    detector.set_input_shape(
        (1, 3, DETECTION_WIDTH, DETECTION_WIDTH)
    )
    detector.set_input_blob_shape((1, 2), id_image_shape)

    results = recognize_from_image(img, detector, args, category)

    logger.info('Script finished successfully.')

    return results

def recebe_imagem(img_path):
    
    detected_people = detect_people(img_path)

    all_results = {} 
    for i in range(0, len(detected_people)):

        img = detected_people[i]

        weight_path, model_path = DATASETS_MODEL_PATH['df2']
        category = DATASETS_CATEGORY['df2']

        parser = get_base_parser(
            'Clothing detection model'
        )
        args = update_parser(parser)

        results_df2 = detect_clothes(img, weight_path, model_path, args, category)
            
        weight_path, model_path = DATASETS_MODEL_PATH['modanet']
        category = DATASETS_CATEGORY['modanet']

        results_modanet = detect_clothes(img, weight_path, model_path, args, category)

        results = results_df2 + results_modanet
        if 'trousers' in results and 'pants' in results:
            results.remove('trousers')

        if 'outer' in results:
            if 'long sleeve outwear' in results or 'long sleeve top' in results:
                results.remove('outer')

        if 'dress' in results:
            if 'long sleeve dress' in results or 'short sleeve dress' in results:
                results.remove('dress')

        if 'dress' in results:
            if 'vest dress' in results:
                results.remove('vest dress')
            if 'sling dress' in results:
                results.remove('sling dress')

        if 'top' in results:
            if 'short sleeve top' in results or 'long sleeve top' in results:
                results.remove('top')

        results = list(filter(lambda x: x != 'footwear', results))
        results = list(filter(lambda x: x != 'belt', results))
        results = list(filter(lambda x: x != 'bag', results))
        results = list(filter(lambda x: x != 'boots', results))
        results = list(filter(lambda x: x != 'sunglasses', results))
        results = list(filter(lambda x: x != 'headwear', results))
        results = list(filter(lambda x: x != 'sling', results))

        results = set(results)

        all_results[i] = results

    return all_results

def weather_predictor(results): 
    print(results)
    clothes_values = {
        'short sleeve top': 0.7,
        'long sleeve top': 0.1,
        'short sleeve outwear': 0.5,
        'long sleeve outwear': 0.1,
        'vest': 0.5,
        'shorts': 0.8,
        'trousers': 0.5,
        'skirt': 0.7,
        'short sleeve dress': 0.7,
        'long sleeve dress': 0.3,
        'vest dress': 0.5,
        'sling dress': 0.5,
        'outer' : 0.2,
        'dress' : 0.7,
        'top' : 0.5,
        'pants' : 0.5,
        'scarf/tie' : 0.1
    }

    soma = 0
    total_clothes = 0  # Contador para o total de peças de roupa consideradas
    for result in results.values():  # Acessa os conjuntos de roupas em cada resultado
        for clothe in result:  # Itera sobre cada peça de roupa no conjunto
            if clothe in clothes_values:  # Verifica se a peça de roupa está no dicionário
                soma += clothes_values[clothe]
                total_clothes += 1  # Aumenta o contador para cada peça válida

    if total_clothes > 0:
        media = soma / total_clothes
        
    if media >= 0.7:
        return 'Quente com temperatura de ' +str(media)+ ' graus Isa'
    elif media >= 0.5 and media < 0.7:
        return 'Ameno com temperatura de ' +str(media)+ ' graus Isa'
    else:
        return 'Frio com temperatura de ' +str(media)+ ' graus Isa'

def main():
    pass

if __name__ == '__main__':
    main()
