# VITA-Audio: 高效大型语音语言模型的快速交错跨模态令牌生成

<p align="center">
    <img src="asset/vita-audio_logo.jpg" width="60%" height="60%">
</p>

<font size=7><div align='center' > [[📖 VITA-Audio 论文](https://arxiv.org/abs/2505.03739)] [[🤖 模型权重](https://huggingface.co/collections/VITA-MLLM/vita-audio-680f036c174441e7cdf02575)]  [[💬 微信群 (微信)](./asset/wechat-group.jpg)]</div></font>

<div align="center">
    <a href="./README_EN.md">🇺🇸 English Version</a> | <a href="./README.md">🇨🇳 中文版本</a>
</div>

## :fire: 最新消息

* **`2025.05.07`** 🌟 我们很自豪地发布VITA-Audio，这是一个具有快速音频-文本令牌生成能力的端到端大型语音模型。

## 📄 目录

- [亮点特性](#-亮点特性)
- [效果展示](#-效果展示)
- [模型介绍](#-模型介绍)
- [实验结果](#-实验结果)
- [环境要求与安装](#-环境要求与安装)
- [训练指南](#-训练指南)
- [推理使用](#-推理使用)
- [评估测试](#-评估测试)

## ✨ 亮点特性

- **低延迟**: VITA-Audio是第一个能够在初始前向传播过程中生成音频的端到端语音模型。通过使用32个预填充令牌，VITA-Audio将生成第一个音频令牌块所需的时间从236毫秒减少到仅53毫秒。
- **快速推理**: VITA-Audio在7B参数规模下实现了3-5倍的推理加速。
- **开源数据**: VITA-Audio仅使用**开源数据**训练，包含20万小时的公开可用音频。
- **强大性能**: VITA-Audio在ASR、TTS和SQA基准测试中，在7B参数以下的尖端模型中取得了具有竞争力的结果。

## 📌 效果展示

### 推理加速
不同推理模式下的模型推理速度。

<p align="center">
  <img src="./asset/qa_speed.gif" alt="问答速度演示" width="48%" style="display: inline-block; margin-right: 2%;">
  <img src="./asset/tts_speed.gif" alt="TTS速度演示" width="48%" style="display: inline-block;">
</p>

### 流式推理中生成第一个音频段的时间
<div align="center">
  <img width="400" alt="第一个音频生成时间" src="https://github.com/user-attachments/assets/165f943e-ac53-443f-abba-e5eb1e0c0f40" />
</div>

### 生成音频案例

> 打南边来了个哑巴，腰里别了个喇叭；打北边来了个喇嘛，手里提了个獭犸。  
> 提着獭犸的喇嘛要拿獭犸换别着喇叭的哑巴的喇叭；别着喇叭的哑巴不愿拿喇叭换提着獭玛的喇嘛的獭犸。  
> 不知是别着喇叭的哑巴打了提着獭玛的喇嘛一喇叭；还是提着獭玛的喇嘛打了别着喇叭的哑巴一獭玛。  
> 喇嘛回家炖獭犸；哑巴嘀嘀哒哒吹喇叭。

https://github.com/user-attachments/assets/38da791f-5d72-4d9c-a9b2-cec97c2f2b2b

---

> To be or not to be--to live intensely and richly,
> merely to exist, that depends on ourselves. Let widen and intensify our relations.   
> While we live, let live!  

https://github.com/user-attachments/assets/fd478065-4041-42bd-9f17-7935b2285799

---

> The hair has been so little, don't think about it, go to bed early, for your hair. Good night!

https://github.com/user-attachments/assets/4cfe4742-e237-42bd-9f17-7935b2285799

---
> 两个黄鹂鸣翠柳，
> 一行白鹭上青天。  
> 窗含西岭千秋雪，
> 门泊东吴万里船。

https://github.com/user-attachments/assets/382620ee-bb2a-488e-9e00-71afd2342b56

## :label: 待办事项

- [x] 发布训练代码和推理代码
- [x] 发布检查点权重
- [x] 发布VITA-Audio-Plus
- [ ] 发布清理后的开源数据JSON和音频

## 🔔 模型介绍

| 模型                   | LLM大小 | Huggingface权重                                           |
|-------------------------|----------|---------------------------------------------------------------|
| VITA-Audio-Boost        | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Boost             |
| VITA-Audio-Balance      | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Balance           |
| VITA-Audio-Plus-Vanilla | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Vanilla      |
| VITA-Audio-Plus-Boost   | 7B       | https://huggingface.co/VITA-MLLM/VITA-Audio-Plus-Boost        |

## 📈 实验结果

- **语音问答对比**

![Clipboard_Screenshot_1746531780](https://github.com/user-attachments/assets/3adcad15-0333-4b92-be49-5a0ace308f3f)

- **文本转语音对比**

![image](https://github.com/user-attachments/assets/09cf8fd3-d7a5-4b77-be49-5a0ace308f3f)

- **自动语音识别对比**

![Clipboard_Screenshot_1746532039](https://github.com/user-attachments/assets/d950cae0-c065-4da9-b37a-a471d28158a0)

![Clipboard_Screenshot_1746532022](https://github.com/user-attachments/assets/929f45cd-693a-4ff6-af73-ceec6e875706)

- **推理加速效果**

![Clipboard_Screenshot_1746532167](https://github.com/user-attachments/assets/ad8b9e90-cd3c-4968-8653-998811a50006)

![Image](https://github.com/user-attachments/assets/4aa5db8c-362d-4152-8090-92292b9a84c0)

## 📔 环境要求与安装

### 准备环境
```bash
docker pull shenyunhang/pytorch:24.11-py3_2024-1224
```

### 获取代码
```bash
git clone https://github.com/VITA-MLLM/VITA-Audio.git
cd VITA-Audio
git submodule update --init --recursive
pip install -r requirements_ds_gpu.txt
pip install -e .
```

### 准备预训练权重

#### LLM模型

- 从 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct 下载LLM模型
- 将其放入 `../models/Qwen/Qwen2.5-7B-Instruct/` 目录

#### 音频编码器和音频解码器

- 从 https://huggingface.co/THUDM/glm-4-voice-tokenizer 下载音频编码器
- 将其放入 `../models/THUDM/glm-4-voice-tokenizer` 目录

- 从 https://huggingface.co/THUDM/glm-4-voice-decoder 下载音频解码器
- 将其放入 `../models/THUDM/glm-4-voice-decoder` 目录

### 数据格式

#### **语音问答数据格式**

```jsonc
{
  "messages": [
    {
      "content": "<|audio|>",
      "role": "user"
    },
    {
      "content": "好的，这样排列更合理：这些生物废弃物如鸡蛋壳、蛤壳、贻贝壳比其他工业废渣更有价值。研究表明，它们在能源、材料、环境保护等领域有广泛应用。高效利用贝壳能提高资源利用效率，减少废弃物，减轻环境负担。特别是在这些领域中，鸡蛋壳因为含有丰富的钙元素，被用于制造医药品和肥料。\n<|audio|>",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/VITA-MLLM/AudioQA-1M/QA_1450K_question_tar/question_shuf_part_8/wav/000000200014510ac1fd776006fc66b36f7f3cda76_question.wav",
    "datasets/VITA-MLLM/AudioQA-1M/QA_1450K_answer_part1_tar/answer_part1_shuf_part_3/wav/000000200114510ac1fd776006fc66b36f7f3cda76_F10.wav"
  ]
}
```

#### **ASR数据格式**

```jsonc
{
  "messages": [
    {
      "content": "Convert the speech to text.\n<|audio|>",
      "role": "user"
    },
    {
      "content": "没有跟大家说是在做什么",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/wenet-e2e/wenetspeech/data/cuts_L_fixed.00000000/X00/X0000016296_135343932_S00019.wav"
  ]
}
```

#### **TTS数据格式**

```jsonc
{
  "messages": [
    {
      "content": "Convert the text to speech.\n那我情愿无药可救。",
      "role": "user"
    },
    {
      "content": "<|audio|>",
      "role": "assistant"
    }
  ],
  "audios": [
    "datasets/Wenetspeech4TTS/WenetSpeech4TTS/Premium/WenetSpeech4TTS_Premium_9/wavs/X0000001735_50639692_S00035.wav"
  ]
}
```

## 🎲 训练指南

以下教程将以 `VITA-Audio-Boost` 为例进行说明。

- 要训练 `VITA-Audio-Balance` 和其他变体，你应该修改 `text-audio-interval-ratio` 参数。

  VITA-Audio-Boost:
  ```bash
  --text-audio-interval-ratio 1 10 4 10 \
  ```

  VITA-Audio-Balance:
  ```bash
  --text-audio-interval-ratio 1 4 3 8 4 10 \
  ```

- 要训练 `VITA-Audio-Plus-*`，你应该使用类似 `scripts/deepspeed/sts_qwen25/finetune_sensevoice_glm4voice...` 的脚本

### 第一阶段 (音频-文本对齐)

```bash
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

上述脚本可能需要一些调整：

- 将 `ROOT_PATH` 设置为你的代码根目录
- 将 `LOCAL_ROOT_PATH` 设置为临时代码根目录
- 根据你的环境修改其他变量

### 第二阶段 (单MCTP模块训练)

```bash
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp1_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

上述脚本可能需要一些调整：

- 将 `ROOT_PATH` 设置为你的代码根目录
- 将 `LOCAL_ROOT_PATH` 设置为临时代码根目录
- 将 `MODEL_NAME_OR_PATH` 设置为第一阶段训练的模型路径
- 根据你的环境修改其他变量

### 第三阶段 (多MCTP模块训练)

```bash
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage1.sh 8192 `date +'%Y%m%d_%H%M%S'`
```

上述脚本可能需要一些调整：

- 将 `ROOT_PATH` 设置为你的代码根目录
- 将 `LOCAL_ROOT_PATH` 设置为临时代码根目录
- 将 `MODEL_NAME_OR_PATH` 设置为第二阶段训练的模型路径
- 根据你的环境修改其他变量

### 第四阶段 (监督微调)

```bash
bash scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh 2048 `date +'%Y%m%d_%H%M%S'`
```

上述脚本可能需要一些调整：

- 将 `ROOT_PATH` 设置为你的代码根目录
- 将 `LOCAL_ROOT_PATH` 设置为临时代码根目录
- 将 `MODEL_NAME_OR_PATH` 设置为第三阶段训练的模型路径
- 根据你的环境修改其他变量

## 📐 推理使用

这里我们实现了一个简单的推理脚本。

它包含了语音转语音、ASR和TTS任务的示例，以及流式和非流式推理速度测试。

```bash
python tools/inference_sts.py
```

- 将 `model_name_or_path` 设置为VITA-Audio权重路径
- 将 `audio_tokenizer_path` 设置为音频编码器路径
- 将 `flow_path` 设置为音频解码器路径

## 🔎 评估测试

评估SQA、ASR和TTS基准测试
```bash
bash scripts/deepspeed/evaluate_sts.sh
```

## &#x1F4E3; 声明

**VITA-Audio在大规模开源语料库上训练，其输出具有随机性。VITA-Audio生成的任何内容都不代表模型开发者的观点。我们对因使用、滥用和传播VITA-Audio而产生的任何问题不承担责任，包括但不限于舆论风险和数据安全问题。**

## :black_nib: 引用

如果你发现我们的工作对你的研究有帮助，请考虑引用以下BibTeX条目。

```bibtex
@misc{,
      title={VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model}, 
      author={Zuwei Long and Yunhang Shen and Chaoyou Fu and Heting Gao and Lijiang Li and Peixian Chen and Mengdan Zhang and Hang Shao and Jian Li and Jinlong Peng and Haoyu Cao and Ke Li and Rongrong Ji and Xing Sun},
      year={2025},
      eprint={2505.03739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.03739}, 
}
```
