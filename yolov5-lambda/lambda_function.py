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
    print('context')
    print(context)
    print(help(context))

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
        conf_thres= int(event['ConfidenceThreshold']),
        iou_thres= int(event['IouThreshold'])
    )

    pred = detect(**params)
    payload_size = sys.getsizeof(pred)
    print(f"Size of response payload: {payload_size}")
    return pred  # Echo back the first key value
    #raise Exception('Something went wrong')

# doesn't run in lambda, just useful for debugging. 
if __name__ == '__main__':
    print('we are in main')
    lambda_handler(json.loads(f"""{{"source": "{DEFAULT_SOURCE}", 
                                "weights": "{DEFAULT_WEIGHTS}", 
                                "is_path": "True"}}"""), None)
    print('finishing main')
