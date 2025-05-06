# VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model

<p align="center">
    <img src="asset/VITA_audio_logos.png" width="50%" height="50%">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.05177" target="_blank"><img src="https://img.shields.io/badge/VITA%20Audio-Report-b5212f.svg?logo=arxiv" /></a>
    <a href="https://huggingface.co/collections/VITA-MLLM/vita-audio-680f036c174441e7cdf02575" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" /></a>
 </p>


## :fire: News



* **`2025.05.06`** 🌟 We are proud to launch VITA-Audio, an end-to-end large speech model with fast audio-text token generation.


## 📄 Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Exhibition](#-exhibition)
- [Models](#-models)
- [Experimental Results](#-experimental-results)
- [Training](#-training)
- [Inference](#-inference)
- [Evaluation](#-evaluation)


## ✨ Highlights

- **Low Latency**. VITA-Audio is the first end-to-end speech model capable of generating audio during the initial forward pass. By utilizing a set of 32 prefill tokens, VITA-Audio reduces the time required to generate the first audio token chunk from 217 ms to just 47 ms.
- **Fast Inference**. VITA-Audio achieves an inference speedup of 3-5x at the 7B parameter scale.
- **Open Source**. VITA-Audio is trained on **open-source data** only, consisting of 200k hours of publicly available audio.
- **Strong Performance**. VITA-Audio achieves competitive results on ASR,TTS and SQA benchmarks among cutting-edge models under 7B parameters.
  


## 📌 Exhibition

### Inference Acceleration
Model inference speed under different inference modes.

<p align="center">
  <img src="./asset/qa_speed.gif" alt="demogif" width="48%" style="display: inline-block; margin-right: 2%;">
  <img src="./asset/tts_speed.gif" alt="second_gif" width="48%" style="display: inline-block;">
</p>

### Time to Generate the First Audio Segment In Streaming Inference
<div align="center">
  <img width="400" alt="first audio generate time" src="https://github.com/user-attachments/assets/165f943e-ac53-443f-abba-e5eb1e0c0f40" />
</div>


### Generated Audio Case



> 打南边来了个哑巴，腰里别了个喇叭；打北边来了个喇嘛，手里提了个獭犸。  
> 提着獭犸的喇嘛要拿獭犸换别着喇叭的哑巴的喇叭；别着喇叭的哑巴不愿拿喇叭换提着獭玛的喇嘛的獭犸。  
> 不知是别着喇叭的哑巴打了提着獭玛的喇嘛一喇叭；还是提着獭玛的喇嘛打了别着喇叭的哑巴一獭玛。  
> 喇嘛回家炖獭犸；哑巴嘀嘀哒哒吹喇叭。

https://github.com/user-attachments/assets/38da791f-5d72-4d9c-a9b2-cec97c2f2b2b


---

> To be or not to be--to live intensely and richly,
> merely to exist, that depends on ourselves. Let widen and intensify our relations.   
> While we live, let live!  

https://github.com/user-attachments/assets/fd478065-4041-4eb8-b331-0c03b304d853


---

> The hair has been so little, don't think about it, go to bed early, for your hair. Good night!

https://github.com/user-attachments/assets/4cfe4742-e237-42bd-9f17-7935b2285799


---
> 两个黄鹂鸣翠柳，
> 一行白鹭上青天。  
> 窗含西岭千秋雪，
> 门泊东吴万里船。

https://github.com/user-attachments/assets/382620ee-bb2a-488e-9e00-71afd2342b56


---
## 🔔 Models

| Model                   | LLM Size | Huggingface Weights                                           |
|-------------------------|----------|---------------------------------------------------------------|
| VITA-Audio-Boost        | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Boost             |
| VITA-Audio-Balance      | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Balance           |
| VITA-Audio-Plus-Vanilla | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla      |



## 📈 Experimental Results
- **Comparison of Spoken Question Answering**.

![Clipboard_Screenshot_1746531780](https://github.com/user-attachments/assets/3adcad15-0333-4b92-bfdf-b753b330a3e2)


- **Comparison of Text to Speech**.

![image](https://github.com/user-attachments/assets/09cf8fd3-d7a5-4b77-be49-5a0ace308f3f)


- **Comparison of Automatic Speech Recognition**.

![Clipboard_Screenshot_1746532039](https://github.com/user-attachments/assets/d950cae0-c065-4da9-b37a-a471d28158a0)

![Clipboard_Screenshot_1746532022](https://github.com/user-attachments/assets/929f45cd-693a-4ff6-af73-ceec6e875706)



- **Effectiveness of Inference Acceleration**.


![Clipboard_Screenshot_1746532167](https://github.com/user-attachments/assets/ad8b9e90-cd3c-4968-8653-998811a50006)

![Image](https://github.com/user-attachments/assets/4aa5db8c-362d-4152-8090-92292b9a84c0)



## 📔 Requirements and Installation

### Prepare Environment
```
docker pull shenyunhang/pytorch:24.11-py3_2024-1224
```

### Get the Code
```
git clone https://github.com/VITA-MLLM/VITA-Audio.git
cd VITA-Audio
pip install -r requirements_ds_gpu.txt
pip install -e .
```

### Prepare Pre-trained Weight

#### LLM

- Download the LLM from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct.
- Put it into '../models/Qwen/Qwen2.5-7B-Instruct/'

#### Audio Encoder and Audio Decoder

- Download the Audio Encoder from https://huggingface.co/THUDM/glm-4-voice-tokenizer.
- Put it into '../models/THUDM/glm-4-voice-tokenizer'

- Download the Audio Decoder from https://huggingface.co/THUDM/glm-4-voice-decoder.
- Put it into '../models/THUDM/glm-4-voice-decoder'


### Data Format
#### **Speech QA Interleaved Data Format**

> This format shows how text and audio sequences are interleaved in a structured JSON conversation between a user and an assistant.

```jsonc
{
  "messages": [
    {
      "role": "user",
      "content": "<|begin_of_audio|> audio_sequence <|end_of_audio|>"
    },
    {
      "role": "assistant",
      "content": "text_sequence_1 <|begin_of_audio|> audio_sequence_1 <|end_of_audio|> text_sequence_2 <|begin_of_audio|> audio_sequence_2 <|end_of_audio|>"
    }
  ]
}
```

## 🎲 Training


The following tutorial will take `VITA-Audio-Boost` as an example.

- To train `VITA-Audio-Balance` and other variants, you should modify the `text-audio-interval-ratio`.

  VITA-Audio-Boost:
  ```
  --text-audio-interval-ratio 1 10 4 10 \
  ```

  VITA-Audio-Balance:
  ```
  --text-audio-interval-ratio 1 4 3 8 4 10 \
  ```

- To train `VITA-Audio-Plus-*`, you should use the script like `scripts/deepspeed/sts_qwen25/finetune_sensevoice_glm4voice...`

### Stage-1 (Audio-Text Alignment)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Modify other variables as needed for your environment.

### Stage-2 (Single MCTP Module Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp1_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 1.
- Modify other variables as needed for your environment.

### Stage-3 (Multiple MCTP Modules Training)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 2.
- Modify other variables as needed for your environment.

### Stage-4 (Supervised Fine-tuning)

```
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh 2048 `date +'%Y%m%d_%H%M%S'`
```

The above script may need some adjustments.

- Set `ROOT_PATH` to your code root folder.
- Set `LOCAL_ROOT_PATH` to a temporary code root folder.
- Set `MODEL_NAME_OR_PATH` to the path of the model trained in Stage 3.
- Modify other variables as needed for your environment.



## 📐 Inference

Here we implement a simple script for inference.

It includes examples of speech-to-speech, ASR, and TTS tasks, as well as inference speed testing.

```
python tools/inference_sts.py
```

- Set `model_name_or_path` to VITA-Audio weights.
- Set `audio_tokenizer_path` to the path of the audio encoder.
- Set `flow_path` to the path of the audio decoder.


## 🔎 Evaluation

Evaluate SQA, ASR, and TTS benchmarks
```
bash scripts/deepspeed/evaluate_sts.sh
```


