
from roboflow import Roboflow
from IPython.display import Image

# requirements to download
"""
!git clone https://github.com/SkalskiP/yolov9.git
!%cd yolov9
!pip install -r requirements.txt -q
!pip install -q roboflow

!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
"""
"""
rf = Roboflow(api_key="KM6S86T8Qr7dLPQ83Gl8")
project = rf.workspace("advanced-honors-cs").project("krunker-player-localization")
version = project.version(1)
dataset = version.download("yolov9")
"""

"""
For training
%cd {HOME}/yolov9

!python train.py \
--batch 16 --epochs 1 --device 0 --min-items 0 --close-mosaic 15 \
--data Krunker-Player-Localization-1/data.yaml \
--weights weights/yolov9-c.pt \
--cfg models/detect/yolov9-c.yaml \
--hyp hyp.scratch-high.yaml
"""

# checking model logs
Image(filename=f"/yolov9/runs/train/exp/results.png", width=1000)