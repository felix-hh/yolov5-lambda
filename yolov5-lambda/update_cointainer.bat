docker build -t yolov5-lambda .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 399705378036.dkr.ecr.us-east-1.amazonaws.com
docker tag  yolov5-lambda:latest 399705378036.dkr.ecr.us-east-1.amazonaws.com/yolov5-lambda:testy
docker push 399705378036.dkr.ecr.us-east-1.amazonaws.com/yolov5-lambda:testy        
aws lambda update-function-code --region us-east-1 --function-name yolov5 --image-uri "399705378036.dkr.ecr.us-east-1.amazonaws.com/yolov5-lambda:testy"
./bell.bat
timeout 2