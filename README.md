# yolo-v3-ncs2-docker
REST API of Yolo V3 using OpenVINO VPU (MYRIAD NCS2) for HW accelerated object recognition.

This project launch a full REST service for running YoloV3 image object recognition on a Intel® Neural Compute Stick 2
It is fully Dockerized, so just launch and use.

https://software.intel.com/en-us/neural-compute-stick

YOLOv3 MYRIAD OpenVINO Docker Container
Run 'docker run --privileged --volume=/dev:/dev <imagename>
Connect to http://<ip>:5000


In QNAP you must create the container from the shell. UI won't work as --volume=/dev:/dev is important! (QNAP v4.3.6.0993 (04-07-2019) on a TS-853A)
docker create --privileged --volume=/dev:/dev --name yolo_rest_openvino --restart=always -p 5000:5000 restserver
Then in QNAP UI, start new container -> settings -> autostart=true -> untick "restart on save" -> save.
You cannot make changes to container in UI. It will violate the /dev:/dev mapping which is required by OpenVINO.


```
 Input JSON:

 {
   "return_original_image": false,     # optional. Returns original image after inference
   "return_marked_image": false,       # optional. Returns a marked image after inference
   "image": "base64 img",              # required. Input image to infer on. Must be base64 encoded and in a format readable by OpenCV (png, bmp, jpg,..). (is optional if return_known_classes = true)
   "image_id": <anytype>               # optional. Non-used type which is passed back in output. Useful for parallel flow-type programming
   "classes": [                        # optional. Array of classes to detect. Set to None or unset to detect all known classes.
       "person",                       #  e.g. detect 'person' at default threshold
       {
           "class": "car",             # Required. e.g. detect 'car'. Could also be class_id (int)
           "threshold": 0.8            #   optional: threshold
           "colorbox": [R,G,B]         #   optional: see global colorbox
           "colortext": [R,G,B]        #   optional: see global colortext
       },
       1                               #  e.g. detect class_id 2
   ],
   "return_known_classes": false,      # optional. Return an array of [class_name, class_name, ...] for all known classes with position = class_id. (is required if image is None)
   "predict_threshold": 0.6,           # optional. Prediction threshold (global). Is also lower-bound. Cannot be less than 0.1.
   "intersection_threshold": 0.4,      # optional. Intersection threshold.
   "debug": 0                          # optional. Increase console output. 0: disable, 1:enable, 2:full output (don't ... long strings to console)
   "colorbox": [255,0,255]             # optional. Global RGB value to draw box.
   "colortext": [0,0,0]                # optional. Global RGB value to draw text.
 }


 Output JSON:
 {
     "org_image": "",                  # original input image. Useful where this machine is part of a flow
     "image_id": <anytype>             # original 'image_id'. Useful for parallel flow-type programming 
     "mark_image": "",                 # A marked image with class matches. In base64/jpeg format 
     "predict_time": 0.5,              # Number of seconds as a float used to predict image
     "known_classes": [..]             # String array of known classes, returned if return_known_classes is set true. Position is equal class_id.
     "match": {                        # OrderedDict of class_name:class match objects. Sorted after highest confidence
         "person": {
             "class_id": 0,            # class_id of match
             "class_name": "person",   # class_name of match
             "confidence": 0.8,        # confidence of match
             "threshold": 0.6,         # threshold required for match
             "left": 0,                # xmin (left) rect of match-box
             "top": 0,                 # ymin (top) rect of match-box
             "right": 0,               # xmax (right) rect of match-box
             "bottom": 0               # ymax (bottom) rect of match-box
         },
         ...
     },
     "predict_threshold": 0.6,         # Prediction threshold used
     "intersection_threshold": 0.4     # intersection threshold used
 }
```

