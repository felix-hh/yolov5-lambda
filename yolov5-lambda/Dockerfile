FROM public.ecr.aws/lambda/python:3.8

COPY . /var/task

WORKDIR /var/task/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html --no-cache-dir
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["lambda_function.lambda_handler"]  