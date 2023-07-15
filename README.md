# Distillation Curriculum Switch for Semantic Segmentation


## 1. prepare the dataset and environment

### 1.1 put the cityscapes on folder ./cityscapes

```
cityscapes
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

### 1.2 conda environment

```
conda env create -f RDD.yaml
```


## 2. run the scripts

### 2.1 train the baseline

```
bash train_scripts/exp_table_1/train_baseline_onestage.sh
```
### 2.2 train RDD

```
bash train_scripts/exp_table_1/train_different_student_two_stage.sh
```

### 2.3 Ablation Study about the iteations of warm-up

```
bash train_scripts/exp_table_4/train_different_wamup_iteration.sh
```

### 2.4 other KD method integrated with RDD

```
# train original KD methods
bash train_scripts/exp_table_5/train_otherKD_onestage.sh

# train KD methods with RDD
bash train_scripts/exp_table_5/train_otherKD_twostage.sh
```
