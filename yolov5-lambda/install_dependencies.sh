rm -r ./env
mkdir ./env
python3 -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html --no-cache-dir --target ./env
python3 -m pip install -r requirements.txt --target ./env --no-cache-dir --upgrade
