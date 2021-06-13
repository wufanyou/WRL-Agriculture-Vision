# WRL-Agriculture-Vision
This repository is the Team WRL's solution to the Agriculture-Vision Prize Challenge.
***

## Generate result with trained models

1.1 Download all checkpoints in models/README.md and put all checkpoints in the model folder

1.2 Modify path in all configs. Check config/README.md for details. 
The default path for dataset is `./supervised/Agriculture-Vision-2021/` 
and `./raw/` (for semi-supervised learning only).

1.3 Run `python test.py` and `python test-semi.py`. We can also specify the input
and output dir as well as the number of GPUs that are used.
Check Reproduction of training models sections for more details.

1.4 By default, a directory called `sumbission` will store the submission for supervised track
and a directory called `semi-submission will store the submission for the semi-supervised track.


## Reproduction of solution
### 1. For supervised learning track

1.1 Modify path in all configs. Check config/README.md for details.

1.2 Train those two models which are required 4 X 2080 Ti GPUs:

    python main.py -c config/DeepLabV3Plus-efficientnet-b3.yaml
    python main.py -c config/FPN-efficientnet-b4.yaml 
    
1.3 Train those two models, which are required 8 X 2080 Ti GPUs of two nodes, 
    check [here](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html)
    to set how to train models on more than one node:

    python main.py -c config/DeepLabV3Plus-efficientnet-b5.yaml
    python main.py -c config/FPN-efficientnet-b5.yaml
   
1.4 Run test script:  

    python test.py -o <OUT-PATH>

`<OUT-PATH>` represents the out path of submission, default value is `submission`. 

If 4 GPUs are used, could run code:
    
      python test.py -f0 -t4 \
    & python test.py -f1 -t4 \
    & python test.py -f2 -t4 \
    & python test.py -f3 -t4 

### 2. For semi-supervised learning track

2.1 Train all models in supervised section 1 (supervised learning track)

2.2 Run `python gen-semi-data.py`, check its CLI for more details.

2.3 Run `python gen-semi-label.py`, check its CLI for more details.

2.4 Train all four models again and use `*-semi.yaml` to
train models again use similar command e.g:

    python main.py -c config/DeepLabV3Plus-efficientnet-b3-semi.yaml

2.5 Run test script, the arguments are the same as `test.py` 
while and default value of `<OUT-PATH>` is set to `semi-submission`:  
    
    python test-semi.py
    
***

## Touble Shooting

Train models with more than one node will need install lastest version of `torchmetrics` from github. The old version is wrongly set-up. check `requirements.txt` for details or use:
   
    pip install git+git://github.com/PyTorchLightning/metrics.git
  


