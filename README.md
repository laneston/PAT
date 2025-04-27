## Reference project

- [A really more real-time adaptation of deep sort](https://github.com/levan92/deep_sort_realtime)
- [YOLOv10: Real-Time End-to-End Object Detection [NeurIPS 2024]](https://github.com/THU-MIG/yolov10)




## Dependencies


the torch [download link is here](https://download.pytorch.org/whl/torch_stable.html)


```
torch==torch-2.3.1 cu118-cp39-cp39-win_amd64.whl
torchvision==torchvision-0.18.1+cu118-cp39-cp39-win_amd64.whl
```

use the following cmd to install torch.

The installation instructions are as follows.

- `pip install '.\torch-2.3.1+cu118-cp39-cp39-win_amd64.whl'`
- `pip install '.\torchvision-0.18.1+cu118-cp39-cp39-win_amd64.whl'`



```
opencv-contrib-python==4.11.0.86
psutil==7.0.0
torchview==0.2.6
```

The installation instructions are as follows.

- numpy, `pip install "numpy<2"`
- opencv, `pip install opencv-contrib-python`
- psutil, `pip install psutil`
- yaml, `pip install PyYAML`
- tqdm, `pip install tqdm`
- requests, `pip install requests`
- pandas,`pip install pandas`
- huggingface_hub, `pip install huggingface_hub`

## Tree

```
.
├── LICENSE
├── README.md
├── app
│   └── videocapture.py
├── deep_sort_realtime
│   ├── README.md
│   ├── __init__.py
│   ├── __pycache__
│   ├── deep_sort
│   ├── deepsort_tracker.py
│   ├── embedder
│   └── utils
├── models
│   ├── download_yolov10_wts.sh
│   ├── yolov10m.pt
│   └── yolov10n.pt
└── ultralytics
    ├── __init__.py
    ├── __pycache__
    ├── assets
    ├── cfg
    ├── data
    ├── engine
    ├── hub
    ├── models
    ├── nn
    ├── solutions
    ├── trackers
    └── utils
 ```

 ## Usage

Enter the app directory by `cd app` and execute instructions `python videocapture.py`.
