### Introduction
This repository uses Ultralytics open source object detection model found at https://github.com/ultralytics/yolov5. The specific architecture of YOLOv5 in this repository is YOLOv5s, lightweight, faster to train and provide inference than the bigger versions at the cost of having less weight parameters, which limits the maximum accuracy of the fine tunned model when trained with large datasets. Since the dataset used for training is quite small, this was not a big concern. 

Two new weight files were obtained by training the model with 4000+ examples of medical instruments. The labeled dataset was obtained using images from https://www.kaggle.com/dilavado/labeled-surgical-tools as well as webscraping and manually labelling google images. These can be found on the yolov5-lambda/weights folder. Sample images can be found in the yolov5-lambda/data folder. The original weights trained with the COCO dataset are also provided to offer inference in non-medical instrument applications.

### API
When the lambda function is triggered with a post http request, lambda_function.py executes the lambda_handler function with the provided events and context parameters. Importantly, the events parameter is a json string with a source and a weights property. The source property contains a base64 encoded image and the weights property contains the name of the weights file to be used for inference, which is 'granular', 'uniform' or 'yolov5s'. 

The granular weights detect 3 classes of medical instruments with labels tweezer, scissor and scalpel. The uniform weights detect all medical instruments and labels them as 'medical instrument'. Finally, the yolov5s weights detect 80+ different objects (including persons, giraffes, knifes and kites, for example) and is provided by Ultralytics, trained on the COCO dataset. The lambda function returns a labeled image encoded in base64 and a json with the list of predictions. Each prediction is also a list, where the first 4 values are the bounding box coordinates, the 5th value is the confidence (a 0 to 1 value) and the 6th value is the class of the object, as an integer. 

### SETUP tips
To prepare the runtime environment for lambda, you will need to install all the dependencies in the yolov5-lambda folder in a similar environment (OS) to the lambda function, since dependencies like numpy need to be compiled in a specific way. 

Launch a ec2 t2.micro instance in aws. Ensure that the Python version >= 3.8 and then run
`<python3 -m pip install -r requirements.txt --target ./packages>`
finally, run the zip_and_deploy.py script, and recover the zip file to upload it to AWS S3. 

Scripts for building and deploying the container are in the yolov5-lambda folder. 
