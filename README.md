## This is an offical implementation of aiDropout paper

## Requires

```sh
python >= 3.6
pytorch >= 1.5
numpy
```

## Run

```sh
cd aidropout
python run_dropout.py [sigma] [rate] [times] [type_model] [num_sampling] [temperature] [epoches] [iters]
# python run_dropout.py 1 1 1 B 1 0.01 20 5

cd idropout
python run_dropout.py [sigma] [rate] [times] [type_model] [iters]

cd pvb
python run.py [tau0] [kappa] [M] [times]

cd svb
# svb rho = 1
# svb-pp rho <= 1
python run_svb.py [rho] [times]
```
## Notice
The dataset here is News Aggregator and is used for concept drift setting
