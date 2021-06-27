import sys
sys.path.append('env')
import json
from detect import detect
# run requried imports, set up everything. 
print('Loading lambda function') 

DEFAULT_SOURCE = "data/images/assistant-surgeon-holds-surgical-instruments-medical-tray-78997883.jpg"
DEFAULT_WEIGHTS = "granular"

WEIGHTS_TO_PATHS = {"yolov5s" : 'weights/yolov5s.pt',
                    "granular" : 'weights/granular.pt',
                    "uniform" : 'weights/uniform.pt'}

# main function executed by lambda. it is passed a json event with keys and a context, that is unused. 
def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    print('context')
    print(context)

    print("calling detection")
    pred = detect(source= event['source'], is_path= event['is_path'], weights= WEIGHTS_TO_PATHS[event['weights']])
    print(pred)

    return pred  # Echo back the first key value
    #raise Exception('Something went wrong')

# doesn't run in lambda, just useful for debugging. 
if __name__ == '__main__':
    print('we are in main')
    lambda_handler(json.loads(f"""{{"source": "{DEFAULT_SOURCE}", 
                                "weights": "{DEFAULT_WEIGHTS}", 
                                "is_path": "True"}}"""), None)
    print('finishing main')
