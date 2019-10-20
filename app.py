

import os
import json
from flask import Flask, request, send_file, jsonify
import logging
import time
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys
from argparse import ArgumentParser, SUPPRESS
import numpy as np
from queue import Queue
from timeit import default_timer as timer
import base64
from math import exp as exp
import math
import collections

def setup_logging():
    #log.basicConfig(format="[ %(asctime)s %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[ %(asctime)s %(levelname)s ] %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log
log = setup_logging()

os.chdir(sys.path[0])

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Path to an .xml file with a trained model. Default yolov3.xml", type=str, default="yolov3.xml")
    args.add_argument("-p", "--port", help="Port to listen on. Default 8080", type=int, default=8080)
    args.add_argument("--labels", help="Labels mapping file. Default=coco.names", type=open, default="coco.names")
    return parser
args = build_argparser().parse_args()


def boot_myriad():
    myriad = {}
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # ------------ Load Inference Engine and network files -------------
    log.info("Loading Inference Engine")
    ie = IECore()
    log.info("IR Engine info:")
    log.info("  Available IR Devices:     {}".format(ie.available_devices))
    devices = ie.get_metric(metric_name='AVAILABLE_DEVICES', device_name='MYRIAD')
    if (len(devices)>1):
        ir_device = "MULTI:"
        for device in devices:
            ir_device += "MYRIAD."+device+","
        ir_device=ir_device[:-1]
    else:
        ir_device = "MYRIAD"
    log.info("  Selected Device(s):       {}".format(ir_device))
    verinfo = ie.get_versions('MYRIAD')['MYRIAD']
    log.info("  Myriad Plugin version:    {}.{}".format(verinfo.major, verinfo.minor))
    log.info("  Myriad Build:             {}".format(verinfo.build_number))

    log.info("Loading network files XML [{}] and BIN [{}]".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    net_input_layer_name = next(iter(net.inputs))
    n, c, h, w = net.inputs[net_input_layer_name].shape
    log.info("  Network shape:   [{},{},{},{}]".format(n,c,h,w))
    
    log.info("Loading labels file [{}]".format(args.labels.name))
    labels_map = [x.strip() for x in args.labels]
   
    log.info("Loading network to device... Please wait..")
    exec_net = ie.load_network(network=net, device_name=ir_device, num_requests=4)
    ir_num_requests = exec_net.get_metric('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
    log.info("  Optimal Number of Infer Requests:  {}".format(ir_num_requests))
    log.info("Done loading network")
    
    myriad = {
        'n': n,
        'c': c,
        'h': h,
        'w': w,
        'labels_map': labels_map,
        'net':net,
        'exec_net': exec_net,
        'net_input_layer_name': net_input_layer_name,
    }
    return myriad
myriad = boot_myriad()

app = Flask(__name__, static_url_path='')

class Error(Exception):
    status_code = 400
    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(Error)
def app_errorhandler(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    log.error(error.message)
    return response

@app.before_request
def before_request():
    log.setLevel(logging.INFO)
    log.info("Request received..")

@app.after_request
def after_request(response):
    log.debug(response)
    return response

@app.route('/', methods=['GET'])
def app_get():
    return send_file('index.html')

@app.route('/', methods=['POST'])
def app_post():
    #
    # Input JSON:
    #
    # {
    #   "return_original_image": false,     # optional. Returns original image after inference
    #   "return_marked_image": false,       # optional. Returns a marked image after inference
    #   "image": "base64 img",              # required. Input image to infer on. Must be base64 encoded and in a format readable by OpenCV (png, bmp, jpg,..). (is optional if return_known_classes = true)
    #   "image_id": <anytype>               # optional. Non-used type which is passed back in output. Useful for parallel flow-type programming
    #   "classes": [                        # optional. Array of classes to detect. Set to None or unset to detect all known classes.
    #       "person",                       #  e.g. detect 'person' at default threshold
    #       {
    #           "class": "car",             # Required. e.g. detect 'car'. Could also be class_id (int)
    #           "threshold": 0.8            #   optional: threshold
    #           "colorbox": [R,G,B]         #   optional: see global colorbox
    #           "colortext": [R,G,B]        #   optional: see global colortext
    #       },
    #       1                               #  e.g. detect class_id 2
    #   ],
    #   "return_known_classes": false,      # optional. Return an array of [class_name, class_name, ...] for all known classes with position = class_id. (is required if image is None)
    #   "predict_threshold": 0.6,           # optional. Prediction threshold (global). Is also lower-bound. Cannot be less than 0.1.
    #   "intersection_threshold": 0.4,      # optional. Intersection threshold.
    #   "debug": 0                          # optional. Increase console output. 0: disable, 1:enable, 2:full output (don't ... long strings to console)
    #   "colorbox": [255,0,255]             # optional. Global RGB value to draw box.
    #   "colortext": [0,0,0]                # optional. Global RGB value to draw text.
    # }
    #

    # Output JSON:
    # {
    #     "org_image": "",                  # original input image. Useful where this machine is part of a flow
    #     "image_id": <anytype>             # original 'image_id'. Useful for parallel flow-type programming 
    #     "mark_image": "",                 # A marked image with class matches. In base64/jpeg format 
    #     "predict_time": 0.5,              # Number of seconds as a float used to predict image
    #     "known_classes": [..]             # String array of known classes, returned if return_known_classes is set true. Position is equal class_id.
    #     "match": {                        # OrderedDict of class_name:class match objects. Sorted after highest confidence
    #         "person": {
    #             "class_id": 0,            # class_id of match
    #             "class_name": "person",   # class_name of match
    #             "confidence": 0.8,        # confidence of match
    #             "threshold": 0.6,         # threshold required for match
    #             "left": 0,                # xmin (left) rect of match-box
    #             "top": 0,                 # ymin (top) rect of match-box
    #             "right": 0,               # xmax (right) rect of match-box
    #             "bottom": 0               # ymax (bottom) rect of match-box
    #         },
    #         ...
    #     },
    #     "predict_threshold": 0.6,         # Prediction threshold used
    #     "intersection_threshold": 0.4     # intersection threshold used
    # }

    # Transfer from global myriad object
    labels_map = myriad['labels_map']

    # Get input job
    data = request.get_json(force=True)
    if not data:
        raise Error("unable to parse input as application/json")

    # Parse input job and defaults
    try:
        job = {
            'return_original_image': True if 'return_original_image' not in data else bool(data['return_original_image']),
            'return_marked_image': True if 'return_marked_image' not in data else bool(data['return_marked_image']),
            'image': None if 'image' not in data else data['image'],
            'image_id': None if 'image_id' not in data else data['image_id'],
            'classes': None if 'classes' not in data else data['classes'],
            'return_known_classes': False if 'return_known_classes' not in data else bool(data['return_known_classes']),
            'predict_threshold': 0.6 if 'predict_threshold' not in data else float(data['predict_threshold']),
            'intersection_threshold': 0.4 if 'intersection_threshold' not in data else float(data['intersection_threshold']),
            'debug': 0 if 'debug' not in data else int(data['debug']),
            'colorbox': [255,0,255] if 'colorbox' not in data else data['colorbox'],
            'colortext': [0,0,0] if 'colortext' not in data else data['colortext']
        }
        job_res = {}
    except Exception as e:
        raise Error("json input parse error: "+str(e))
    
    if job['debug']:
        log.info("Turning on DEBUG lvl:"+str(job['debug']))
        log.setLevel(logging.DEBUG)
        out_job = job
        if job['debug'] < 2 and out_job['image']:
            out_job['image'] = out_job['image'][:10]+'...'
        log.debug("New job (/w defaults set):\n{}".format(json.dumps(out_job, indent=4)))

    # Basic checks
    if not job['image'] and not job['return_known_classes']:
        raise Error("No 'image' and return_known_classes is not true, so nothing to do")

    # return original image if requested
    if job['return_original_image']:
        job_res['org_image'] = job['image']

    # return the image_id if requested
    if job['image_id']:
        job_res['image_id'] = job['image_id']

    # return predict_threshold and intersection_threshold as is used in this job
    job_res['predict_threshold'] = job['predict_threshold']
    job_res['intersection_threshold'] = job['intersection_threshold']

    if job['return_known_classes']:
        job_res['known_classes'] = labels_map

    # if no input image but request for know classes, return them now
    if not job['image'] and job['return_known_classes']:
        return job_res

    # Build match_classes dict as ordered (index key must be equal to original labels_map which AI is trained on)
    # result match_classes:
    #   'person': { 'threshold': 0.8, 'colorbox': [1,2,3], 'colortext': [1,2,3]},
    #   'car': ...
    # Get by index (class_id) via >>list(m.items())[class_id]<<
    match_classes = collections.OrderedDict(zip(labels_map, [1]*len(labels_map)))   # can't add {} to this []*len() construct as it would make 1 dict which is referenced. hence any change to subscript would be global
    for match_class in match_classes.keys():
        match_classes[match_class] = {
            'threshold': job['predict_threshold'],
            'colorbox': job['colorbox'],
            'colortext': job['colortext'],
        }
    # If job['classes'] is set, then adjust threshold so that it is:
    #   255 for classes not found in job['classes'] 
    #   keep for classes found in job['classes'] but who does not specifiy threshold
    #   set for classes found in job['classes'] who do specify threshold
    try:
        if job['classes']:
            tmp_det_map = []
            for job_class in job['classes']:
                if isinstance(job_class, str):
                    tmp_det_map += [job_class]
                elif isinstance(job_class, int):
                    tmp_det_map += [labels_map[job_class]]
                elif isinstance(job_class, dict):
                    job_class_class = job_class['class'] if isinstance(job_class['class'], str) else labels_map[job_class['class']]
                    job_class_thre = job_class['threshold'] if 'threshold' in job_class else job['predict_threshold']
                    job_class_colorbox = job_class['colorbox'] if 'colorbox' in job_class else job['colorbox']
                    job_class_colortext = job_class['colortext'] if 'colortext' in job_class else job['colortext']
                    tmp_det_map += [job_class_class]
                    match_classes[job_class_class]['threshold'] = job_class_thre
                    match_classes[job_class_class]['colorbox'] = job_class_colorbox
                    match_classes[job_class_class]['colortext'] = job_class_colortext
                else:
                    raise "'classes' must be an array of str, int or dicts"
            # set threshold -1 for classes not in tmp_det_map
            for match_class in match_classes.keys():
                if match_class not in tmp_det_map:
                    match_classes[match_class]['threshold'] = 255
    except Exception as e:
        raise Error("'classes' passing error: "+str(e))
    
    if job['debug'] > 1:
        log.debug("Match Classes:\n\n{}\n\n".format(match_classes))

    # b64 decode image, load it and resize to network size
    try:
        bin_image = base64.b64decode(job['image'], validate=True)
        bin_image = np.asarray(bytearray(bin_image), dtype="uint8")
        image = cv2.imdecode(bin_image, cv2.IMREAD_COLOR)
        in_image = cv2.resize(image, (myriad['w'], myriad['h']))
        in_image = in_image.transpose((2,0,1)) # change data layout from HWC to CHW
        in_image = in_image.reshape((myriad['n'], myriad['c'], myriad['h'], myriad['w']))
    except Exception as e:
        raise Error("'image' decode error: "+str(e))

    # start inference
    try:
        start_time = timer()
        output = myriad['exec_net'].infer(inputs={myriad['net_input_layer_name']: in_image})
        stop_time = timer()
        inference_time = stop_time - start_time
        job_res['predict_time'] = round(inference_time, 3)
        log.info("Inference done in {} seconds".format(inference_time))
    except Exception as e:
        raise Error("Inference engine error: "+str(e))

    net = myriad['net']
    objects = list()
    for layer_name, out_blob in output.items():
        out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].shape)
        layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        #log.info("Layer {} parameters: ".format(layer_name))
        #layer_params.log_params()
        objects += parse_yolo_region(out_blob, in_image.shape[2:], image.shape[:-1], layer_params, job['predict_threshold'])

    # Filtering overlapping boxes with respect to the intersection_threshold
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > job['intersection_threshold']:
                objects[j]['confidence'] = 0
    
    # Filter objects with respect to the predict_threshold
    objects = [obj for obj in objects if obj['confidence'] >= match_classes[labels_map[obj['class_id']]]['threshold']]

    if len(objects):
        log.info(" Class ID | Confidence | Req.Conf | LEFT | TOP | RIGHT | BOTTOM | COLOR ")

    # loop over each object and pass for output and drawing
    origin_im_size = image.shape[:-1]
    job_res['match'] = []
    for obj in objects:
        class_id = obj['class_id']
        match_class = list(match_classes.items())[class_id]
        match_class_name = match_class[0]
        match_class_threshold = match_class[1]['threshold']
        match_class_colorbox = match_class[1]['colorbox'][::-1] # RBG -> BGR
        match_class_colortext = match_class[1]['colortext'][::-1] # RBG -> BGR
        rectbox = [
            max(obj['xmin'], 0),
            max(obj['ymin'], 0),
            min(obj['xmax'], origin_im_size[1]),
            min(obj['ymax'], origin_im_size[0])
            ]
        # Make output match object
        job_res['is_{}'.format(match_class_name)] = True
        confidence_key = 'is_{}_confidence'.format(match_class_name)
        confidence_val = round(obj['confidence'] * 100, 1)
        job_res[confidence_key] = job_res[confidence_key] if confidence_key in job_res and confidence_val < job_res[confidence_key] else confidence_val
        job_res['match'] += [{
                'class_id': class_id,
                'class_name': match_class_name,
                'confidence': confidence_val,
                'threshold': match_class_threshold,
                'left': rectbox[0],
                'top': rectbox[1],
                'right': rectbox[2],
                'bottom': rectbox[3]
            }]
    #     "match": {                        # OrderedDict of class_name:class match objects. Sorted after highest confidence
    #         "person": {
    #             "class_id": 0,            # class_id of match
    #             "class_name": "person",   # class_name of match
    #             "confidence": 0.8,        # confidence of match
    #             "threshold": 0.6,         # threshold required for match
    #             "left": 0,                # xmin (left) rect of match-box
    #             "top": 0,                 # ymin (top) rect of match-box
    #             "right": 0,               # xmax (right) rect of match-box
    #             "bottom": 0               # ymax (bottom) rect of match-box
    #         },
    #         ...
    #     },

        # Validation bbox of detected object (set to edge if detection box is outside edges)
        det_label = labels_map[class_id] if labels_map and len(labels_map) >= class_id else "#"+str(class_id)
        log.info("{:^9} | {:10f} | {:8} | {:4} | {:3} | {:5} | {:6} | {} ".format(det_label, obj['confidence'], match_class_threshold, rectbox[0], rectbox[1], rectbox[2], rectbox[3], match_class_colorbox))

        if not job['return_marked_image']:
            #don't draw on image if none is to be returned
            continue

        cv2.rectangle(image, (rectbox[0],rectbox[1]), (rectbox[2], rectbox[3]), match_class_colorbox, 2)
        
        # Draw black text on a filled rectangle
        txt = det_label + ' ' + str(confidence_val) + ' %'
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontscale = 0.6
        fontthick = 1
    
        # Get size of text written using font, scale and thickness
        ret, baseline = cv2.getTextSize(txt, font, fontscale, fontthick)


        # calc box for writing text
        txt_rect = (
                rectbox[0]-1,
                rectbox[1] - ret[1] - baseline - 1,
                rectbox[0] + ret[0] + 1,
                rectbox[1] + 1
                )
        # if txt rect is outside image, put txt rect inside draw-rect
        if txt_rect[1] < 0:
            txt_rect = (
                rectbox[0]-1,
                0,
                rectbox[0] + ret[0] + 1,
                0 + ret[1] + baseline + 1
                )

        # draw box and write black text
        cv2.rectangle(image, (txt_rect[0], txt_rect[1]), (txt_rect[2], txt_rect[3]), match_class_colorbox, -1)
        cv2.putText(image, txt, (rectbox[0], txt_rect[1] + ret[1] + 1), font, fontscale, match_class_colortext, 1, cv2.LINE_AA)

    # encode image as jpg and base64 encode it
    if job['return_marked_image']:
        retval, mark_image = cv2.imencode('.jpg', image)
        mark_image = base64.b64encode(mark_image).decode('ascii')
        job_res['mark_image'] = mark_image

    # all done. Return job_res
    return job_res


def test():
    log.info("Running test..")
    with open('/share/img/1_88.jpg', 'rb') as f:
        image = base64.b64encode(f.read())

    app.testing = True
    with app.test_client() as c:
        res = c.post('/', json={
            'return_original_image': False,
            'return_marked_image': True,
            'image': image
            })
    log.info("Test returned [{}]".format(res.get_json()))

if 0:
    test()
else:
    app.run(host='0.0.0.0', port=5000, debug=False)

