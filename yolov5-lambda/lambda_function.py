import json
import sys
from detect import detect
# run requried imports, set up everything. 
print('Loading lambda function') 

DEFAULT_SOURCE = "data/images/assistant-surgeon-holds-surgical-instruments-medical-tray-78997883.jpg"
DEFAULT_WEIGHTS = "granular"

WEIGHTS_TO_PATHS = {"yolov5s" : 'weights/yolov5s.pt',
                    "granular" : 'weights/granular.pt',
                    "uniform" : 'weights/uniform.pt'}

IS_PATH_TRUE_FLAG = "TRUE"
SAVE_TRUE_FLAG = "TRUE"

# main function executed by lambda. it is passed a json event with keys and a context, that is unused. 
def lambda_handler(event, context):
    remaining_time = context.get_remaining_time_in_millis()
    context.log(f"remaining time in milliseconds: {remaining_time}")

    print("calling detection")
    # obtain function parameters
    
    width= int(event['Width'])
    height= int(event['Height'])
    
    params = dict(
        source= event['Source'],
        shape= (height, width),
        is_path= event['IsPath'].upper() == IS_PATH_TRUE_FLAG,
        weights= WEIGHTS_TO_PATHS[event['Weights']], 
        save= event['Save'].upper() == SAVE_TRUE_FLAG,
        imgsz= int(event['InferenceImageSize']),
        conf_thres= float(event['ConfidenceThreshold']),
        iou_thres= float(event['IouThreshold'])
    )

    pred = detect(**params)
    payload_size = sys.getsizeof(pred)
    context.log(f"Size of response payload: {payload_size}")

    remaining_time = context.get_remaining_time_in_millis()
    context.log(f"remaining time in milliseconds: {remaining_time}")

    return pred  # Echo back the first key value
    #raise Exception('Something went wrong')

# sample context for testing
class Context():
    def __init__(self):
        return
    def log(self, message):
        print(message)
        return
    def get_remaining_time_in_millis(self):
        return -1


# doesn't run in lambda, just useful for debugging. 
if __name__ == '__main__':
    print('we are in main')

    context = Context()
    sample_request = """{
            "Source": "./data/images/zidane.jpg",
            "Width": 1080,
            "Height": 810,
            "IsPath": "True",
            "Weights": "yolov5s",
            "Save": "False",
            "InferenceImageSize": 640,
            "ConfidenceThreshold": 0.25,
            "IouThreshold": 0.45
            }"""
    lambda_handler(json.loads(sample_request), context)
    print('finishing main')
