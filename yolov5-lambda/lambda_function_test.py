from lambda_function import *
from pathlib import Path
import base64
from io import BytesIO
import json
from PIL import Image

# sample context for testing
class Context():
    def __init__(self):
        return
    def log(self, message):
        #don't log full images to this console. 
        print(str(message)[:100])
        return
    def get_remaining_time_in_millis(self):
        return -1


INPUT_B64_BUS = Path("./data/base64_test/bus_b64.json")

sample_request = json.loads("""{
        "Source": "./data/images/zidane.jpg",
        "IsPath": "True",
        "Weights": "yolov5s",
        "Save": "true",
        "InferenceImageSize": 640,
        "ConfidenceThreshold": 0.25,
        "IouThreshold": 0.45
        }""")

sample_request_2 = sample_request.copy() #important, copy dict, otherwise you're mutating the original too. 
base_64_data = json.loads(open(INPUT_B64_BUS).read())
image_b64 = base_64_data['image']['data']

sample_request_2['Source'] = image_b64
sample_request_2['IsPath'] = 'False'

def get_image_from_response(response):
    return Image.open(BytesIO(base64.b64decode(response['image'])))

# doesn't run in lambda, just useful for debugging. 
if __name__ == '__main__':
    print('we are in main')
    context = Context()
    print('running first test')
    pred1 = lambda_handler(sample_request, context)
    print('running second test')
    pred2 = lambda_handler(sample_request_2, context)
    print('finishing main')
    print('saving test results')
    test_dir = Path('./runs/testing')
    test_dir.mkdir(exist_ok=True)
    image1 = get_image_from_response(pred1)
    image2 = get_image_from_response(pred2)
    image1.save(test_dir / 'test1.jpg')
    image2.save(test_dir / 'test2.jpg')

    # now get the entire request from a json file. 
    sample_request_3 = json.loads(open('request_base64_example.json').read())
    pred3 = lambda_handler(
        {'body': json.dumps(sample_request_3)}, 
        context)
    image3 = get_image_from_response(pred3)
    image3.save(test_dir / "test3.jpg")

    # and test pinging works
    lambda_handler({'isPing': 'True'}, context)
