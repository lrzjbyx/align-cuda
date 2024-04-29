# import build.libexample
import build.sealnet_align

import cv2
import numpy as np
import json
import time

ss = {
    "la":False,
    "mu":True,
    "x": 83.8255813953487,
    "y": 728.8168604651161,
    "rect": [
        154.0,
        -119.0,
        325.6133720930233,
        10.0
    ],
    "rotation": 0,
    "text": "测试专用章",
    "type": 2,
    "sequence": "从左到右",
    "l": 325.6133720930233,
    "h": 97
}

# ss = {
#     "la":True,
#     "mu":False,
#     "x": 11.64244186046517,
#     "y": 328.3168604651162,
#     "rect": [
#         95.0,
#         -219.0,
#         584.0755813953488,
#         584.0755813953488
#     ],
#     "rotation": 0,
#     "text": "郑州大学档案项目组测试专用章",
#     "sequence": "从左到右",
#     "type": 5,
#     "startAngle": 5152,
#     "spanAngle": 4112,
#     "a": 292.0377906976744,
#     "b": 292.0377906976744,
#     "h": 154
# }

for i in range(10000):
    start_time = time.time()
    image = cv2.imread("image.png")
    result = build.sealnet_align.align(image,ss)
    cv2.imwrite("result.png",result)
    end_time = time.time()
    run_time = end_time - start_time
    print("该程序执行了 {:.2f} 秒".format(run_time))

