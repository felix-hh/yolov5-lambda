import json
import sys
import time
from detect import detect
# run requried imports, set up everything. 
print('Loading lambda function') 

DEFAULT_SOURCE = r"C:\Users\felix\Desktop\yolov5_on_lambda\yolov5-lambda\data\images\drawing_tray.jpg"
DEFAULT_WEIGHTS = "granular"

WEIGHTS_TO_PATHS = {"yolov5s" : 'weights/yolov5s.pt',
                    "granular" : 'weights/granular.pt',
                    "uniform" : 'weights/uniform.pt'}

IS_PATH_TRUE_FLAG = "TRUE"
SAVE_TRUE_FLAG = "TRUE"

# main function executed by lambda. it is passed a json event with keys and a context, that is unused. 
def lambda_handler(event, context):
    context.log('Request received \n')
    # context.log(event)
    body = json.loads(event['body'])
    # dont log the image in the cloud, it is expensive. TODO log the image only in case of errors. 
    # context.log(body)

    # if IsPing it means the request is comming from a CloudWatch scheduled event to keep our Lambda instance warm and fast
    if 'isPing' in body:
        sample_request_3 = json.loads(open('request_base64_example.json').read())
        sample_request_3['Save'] = 'False' # ensure we don't save images every time we ping!
        # there's no need to do recursion to ping detect.py but I'm lazy enough to recycle the test
        pred3 = lambda_handler({
            'body': json.dumps(sample_request_3)
        }, context)
        return f'Lambda pinged at {time.time()}'

    # obtain function parameters
    params = dict(
        source= body['Source'],
        is_path= body['IsPath'].upper() == IS_PATH_TRUE_FLAG,
        weights= WEIGHTS_TO_PATHS[body['Weights']], 
        save= body['Save'].upper() == SAVE_TRUE_FLAG,
        imgsz= int(body['InferenceImageSize']),
        conf_thres= float(body['ConfidenceThreshold']),
        iou_thres= float(body['IouThreshold'])
    )

    context.log("calling detection")
    pred = detect(**params)
    payload_size = sys.getsizeof(pred)
    context.log(f"Size of response payload: {payload_size}\n")

    return pred
    #raise Exception('Something went wrong')

