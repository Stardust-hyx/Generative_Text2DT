# Generative Text2DT
[![GitHub stars](https://img.shields.io/github/stars/Stardust-hyx/Generative_Text2DT?style=flat-square)](https://github.com/Stardust-hyx/Generative_Text2DT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Stardust-hyx/Generative_Text2DT?style=flat-square&color=blueviolet)](https://github.com/Stardust-hyx/Generative_Text2DT/network/members)

Source code for the paper, *Generative models for automatic medical decision rule extraction from text*.

## Quick links

- [Requirements](#requirements)
- [Data](#data)
- [Run](#run)
  - [Sequence-to-sequence Models](#sequence-to-sequence-models)
  - [Autoregressive Models](#autoregressive-models)

## Requirements
```
Python >= 3.7   
PyTorch >= 1.6.0 
Transformers == 3.4.0
openai
zhipuai
```

## Data
Download raw json data from [https://tianchi.aliyun.com/dataset/95414](https://tianchi.aliyun.com/dataset/95414) and place at **./json**. Preprocess the raw data by running
```shell
cd data
python preprocess.py
cd Text2DT
python data_augment.py
python convert.py
```
The preprocessing procedure involves manual intervention, see **data/preprocess.py** for details.

## Run

### Sequence-to-sequence Models
#### For NL-style linearization
```shell
cd Seq2seq_NL
# set a different random seed for each run
bash train_eval.sh
# ensemble predictions by multiple models
python ensemble.py
```

#### For AugNL-style linearization
```shell
cd Seq2seq_AugNL
# set a different random seed for each run
bash train_eval.sh
# ensemble predictions by multiple models
python ensemble.py
```

### Autoregressive Models

#### For JSON-style linearization (ICL setting)
```shell
cd Autoregressive_ICL
# Using ChatGPT
python chatgpt_JSON.py
# Using ChatGLM
python chatglm_JSON.py
```

#### For NL-style linearization (ICL setting)
```shell
cd Autoregressive_ICL
# Using ChatGPT
python chatgpt_NL.py
# Using ChatGLM
python chatglm_NL.py
```

#### For NL-style linearization (SFT setting)
*Note that, please create a new environment following the instruction [here](https://github.com/Stardust-hyx/Instruction_Tuning).*
```shell
cd Autoregressive_SFT
# set a different random seed for each run
bash train_eval.sh
# ensemble predictions by multiple models
python ensemble.py
```
