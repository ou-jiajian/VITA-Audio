# VITA-Audio: Fast Interleaved Audio-Text Token Generation for Efficient Large Speech-Language Model

<p align="center">
    <img src="asset/VITA_audio_logos.png" width="50%" height="50%">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.05177" target="_blank"><img src="https://img.shields.io/badge/VITA%20Audio-Report-b5212f.svg?logo=arxiv" /></a>
    <a href="https://huggingface.co/collections/VITA-MLLM/vita-audio-680f036c174441e7cdf02575" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" /></a>
 </p>


## :fire: News



* **`2025.05.06`** 🌟 We are proud to launch VITA-Audio, an end-to-end large speech model with fast audio-text token generation.


## Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Experimental Results](#-experimental-results)
- [Models](#-models)
- [Training, Inference and Evaluation](#-training-inference-and-evaluation)


## ✨ Highlights

- **Low Latency**. VITA-Audio is the first end-to-end speech model capable of generating audio during the initial forward pass. By utilizing a set of 32 prefill tokens, VITA-Audio reduces the time required to generate the first audio token chunk from 217 ms to just 47 ms.
- **Fast Inference**. VITA-Audio achieves an inference speedup of 3-5x at the 7B parameter scale.
- **Open Source**. VITA-Audio is trained on **open-source data** only, consisting of 200k hours of publicly available audio.
- **Strong Performance**. VITA-Audio achieves competitive results on ASR,TTS and SQA benchmarks among cutting-edge models under 7B parameters.
  



<p align="center" style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px;">
  <img src="./asset/qa_speed.gif" alt="demogif" style="max-width: 100%; height: auto; object-fit: contain; flex: 1;">
  <img src="./asset/another_gif.gif" alt="second_gif" style="max-width: 100%; height: auto; object-fit: contain; flex: 1;">
</p>

## 🐍 Models

| Model              | LLM Size | Huggingface Weights                                      |
|--------------------|----------|----------------------------------------------------------|
| VITA-Audio-Boost   | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Boost        |
| VITA-Audio-Balance | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Balance      |



## 📈 Experimental Results
- **Comparison of Spoken Question Answering**.

![Image](https://github.com/user-attachments/assets/382f4ffa-a2e2-4f03-8c5d-d52626311f3a)



- **Comparison of Automatic Text to Speech**.

![Image](https://github.com/user-attachments/assets/0a49aff6-b94d-4621-97bf-7e7f6ca2c1c5)

- **Comparison of Automatic Speech Recognition**.

![Image](https://github.com/user-attachments/assets/9a89f29c-d96c-4bfe-af0c-3a68bf5f6eee)


- **Effectiveness of Inference Acceleration**.





![Image](https://github.com/user-attachments/assets/be147a04-e04d-4a6b-bde2-374ae8975c58)



![Image](https://github.com/user-attachments/assets/4aa5db8c-362d-4152-8090-92292b9a84c0)




## Requirements and Installation

```
git clone https://github.com/VITA-MLLM/VITA-Audio.git
cd VITA-Audio
pip install -r requirements_ds_gpu.txt
pip install -e .
```

### Prepare Pre-trained Weight

### LLM
- Download the LLM from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct.
- Put it into '../models/Qwen/Qwen2.5-7B-Instruct/'

### Audio Encoder and Audio Decoder
- Download the Audio Encoder from https://huggingface.co/THUDM/glm-4-voice-tokenizer.
- Put it into '../models/THUDM/glm-4-voice-tokenizer'

- Download the Audio Decoder from https://huggingface.co/THUDM/glm-4-voice-decoder.
- Put it into '../models/THUDM/glm-4-voice-decoder'


## Training
### Stage-1 (Audio-Text Alignment)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_stage1.sh 8192 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Modify other variables as needed for your environment.

### Stage-2 (Single MCTP Module Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp1_stage1.sh 8192 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 1.
- Modify other variables as needed for your environment.

### Stage-3 (Multiple MCTP Modules Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage1.sh 8192 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 2.
- Modify other variables as needed for your environment.

### Stage-4 (Supervised Fine-tuning)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh 8192 `date +'%Y%m%d_%H'`0000
```

The above script may need some adjustment.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 3.
- Modify other variables as needed for your environment.



## 📐Inference

We provide the converted Huggingface weights in

- https://huggingface.co/VITA-MLLM/VITA-Audio-Boost 
- https://huggingface.co/VITA-MLLM/VITA-Audio-Balance 


- Set `model_name_or_path` to your VITA-Audio weights.
- Set `audio_tokenizer_path` to the path of the audio encoder.
- Set `flow_path` to the path of the audio decoder.


Here we implement a simple script for inference
```
python tools/inference_sts.py
```


## Evaluation

Evaluate with benchmarks
```
bash scripts/deepspeed/evaluate_sts.sh 1024 `date +'%Y%m%d_%H'`0000
```


