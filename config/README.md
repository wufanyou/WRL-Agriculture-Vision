# Placeholder directory for model configurations

***

Here are description of fields that might need modification in the config file. 
Check comments in the config files for more details. 


| Key      | Description |
| ----------- | ----------- |
|`DATASET.PATH`|The path of dataset|
|`EXPERIMENT.SAVER.DIRPATH`| The path to save model checkpoints |
|`TRAINER.GPUS`| Number of GPUs are used|
|`TRAINER.NUM_NODES`|Number of Nodes are used|
|`TRAINER.MAX_STEPS`|number of iteration|
|`TRAINER.MAX_EPOCHS`|number of epoch|
|`DATALOADER.BATCH_SIZE.TRAIN`| The mini batch size when training|
|`DATALOADER.BATCH_SIZE.VAL`| The mini batch size when validating|
|`OPTIMIZER.ARGS.LR`|Learning rate|
|`MODEL.PRETRAINED`| Pretrained models files. Those models are only used for semi supervised learning|
