import json
import sys
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
    remaining_time = context.get_remaining_time_in_millis()
    context.log(f"remaining time in milliseconds: {remaining_time}\n")

    print("calling detection")
    # obtain function parameters
        
    params = dict(
        source= event['Source'],
        is_path= event['IsPath'].upper() == IS_PATH_TRUE_FLAG,
        weights= WEIGHTS_TO_PATHS[event['Weights']], 
        save= event['Save'].upper() == SAVE_TRUE_FLAG,
        imgsz= int(event['InferenceImageSize']),
        conf_thres= float(event['ConfidenceThreshold']),
        iou_thres= float(event['IouThreshold'])
    )

    pred = detect(**params)
    payload_size = sys.getsizeof(pred)
    context.log(f"Size of response payload: {payload_size}\n")

    remaining_time = context.get_remaining_time_in_millis()
    context.log(f"remaining time in milliseconds: {remaining_time}\n")

    return pred  # Echo back the first key value
    #raise Exception('Something went wrong')

