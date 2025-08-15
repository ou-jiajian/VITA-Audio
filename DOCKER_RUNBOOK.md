# VITA-Audio Docker 环境一键跑通指南

> 本文档提供标准化的 Docker 部署与运行流程，确保在一致环境下完成推理与四阶段训练。后续安装与部署请严格按本文档执行。

## 0. 前置条件
- 已安装 Docker，并可使用 sudo 运行
- 服务器已安装 NVIDIA 驱动，`nvidia-smi` 正常
- 已安装 NVIDIA Container Toolkit（容器内可见 GPU）

自检命令：
```bash
nvidia-smi
sudo docker --version
sudo docker info | grep -i runtime
```
若 `sudo docker info` 输出包含 `Runtimes: nvidia runc`，说明 GPU 容器运行环境可用。

---

## 1. 获取标准镜像
```bash
sudo docker pull shenyunhang/pytorch:24.11-py3_2024-1224
```
若拉取缓慢或失败，可配置镜像加速（可选）：
```bash
# 配置加速源
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
EOF
sudo systemctl restart docker
```

---

## 2. 目录规范与数据映射
训练脚本默认以 `/data/` 为根路径，请按此规范准备宿主机目录并挂载：
```bash
# 宿主机上执行
sudo mkdir -p /data/models /data/output /data/data
# 将项目代码统一映射到 /data/VITA-Audio
sudo ln -sfn /home/linux/VITA/VITA-Audio /data/VITA-Audio
```

---

## 3. 启动容器（启用全部 GPU）
```bash
sudo docker run --gpus all -it --shm-size=16g \
  -v /data:/data \
  --name vita-audio \
  shenyunhang/pytorch:24.11-py3_2024-1224 bash
```
提示：如容器已存在，用 `sudo docker start -ai vita-audio` 进入。

---

## 4. 容器内初始化环境
以下命令均在容器内执行：
```bash
cd /data/VITA-Audio
# 初始化子模块与依赖
git submodule update --init --recursive
pip install -r requirements_ds_gpu.txt
pip install -e .
```

---

## 5. 准备预训练权重
目录要求：
- LLM: `/data/models/Qwen/Qwen2.5-7B-Instruct/`
- 音频编码器: `/data/models/THUDM/glm-4-voice-tokenizer`
- 音频解码器: `/data/models/THUDM/glm-4-voice-decoder`

下载示例（需 `huggingface-cli login` 后使用）：
```bash
mkdir -p /data/models/Qwen /data/models/THUDM
# LLM
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen/Qwen2.5-7B-Instruct
# 音频编码器/解码器
huggingface-cli download THUDM/glm-4-voice-tokenizer \
  --local-dir /data/models/THUDM/glm-4-voice-tokenizer
huggingface-cli download THUDM/glm-4-voice-decoder \
  --local-dir /data/models/THUDM/glm-4-voice-decoder
```

---

## 6. 快速推理自检（强烈建议先跑通）
```bash
cd /data/VITA-Audio
python tools/inference_sts.py
```
包含：文本与流式文本、S2S 与流式 S2S、ASR、TTS（含克隆）、推理速度基准。

默认音频输出：
```
/data/output/LM/inference/asset/.../*.wav
```

如需使用自己训练的权重，编辑 `tools/inference_sts.py` 顶部：
```python
model_name_or_path = "/data/output/LM/.../你的stage4输出目录/"
audio_tokenizer_path = "/data/models/THUDM/glm-4-voice-tokenizer"
flow_path = "/data/models/THUDM/glm-4-voice-decoder"
```

---

## 7. 四阶段训练流程
训练会在 `/data/VITA-Audio/` 进行，输出到 `/data/output/...`。

单机单卡示例：
```bash
export NPROC_PER_NODE=1
```

### 7.1 Stage-1 音频-文本对齐
```bash
bash /data/VITA-Audio/scripts/deepspeed/sts_qwen25/finetune_glm4voice_stage1.sh \
  8192 $(date +'%Y%m%d_%H%M%S')
```
关键变量（脚本内）：
- `ROOT_PATH=/data/`
- `MODEL_NAME_OR_PATH=/data/models/Qwen/Qwen2.5-7B-Instruct/`
- `AUDIO_TOKENIZER_PATH=/data/models/THUDM/glm-4-voice-tokenizer`

### 7.2 Stage-2 单 MCTP
将 `MODEL_NAME_OR_PATH` 指向 Stage-1 的输出目录：
```bash
bash /data/VITA-Audio/scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp1_stage1.sh \
  8192 $(date +'%Y%m%d_%H%M%S')
```

### 7.3 Stage-3 多 MCTP
将 `MODEL_NAME_OR_PATH` 指向 Stage-2 的输出目录：
```bash
bash /data/VITA-Audio/scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage1.sh \
  8192 $(date +'%Y%m%d_%H%M%S')
```

### 7.4 Stage-4 监督微调
将 `MODEL_NAME_OR_PATH` 指向 Stage-3 的输出目录：
```bash
bash /data/VITA-Audio/scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh \
  2048 $(date +'%Y%m%d_%H%M%S')
```

---

## 8. 评估
```bash
bash /data/VITA-Audio/scripts/deepspeed/evaluate_sts.sh
```

---

## 9. 常见问题（FAQ）
- 容器内看不到 GPU：
  ```bash
  nvidia-smi
  # 若无输出，确认主机已安装 nvidia-container-toolkit 并重启 docker
  ```
- 拉取镜像超时：按第 1 节配置镜像加速后重试
- 权重下载慢或失败：
  - 先在可联网环境下载，再通过挂载目录映射到容器内
  - 使用 `huggingface-cli login` 登录后再下载
- 显存不足（2080Ti/单卡）：降低 batch、增大梯度累积，例如：
  ```
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  ```
- flash-attn 报错：GPU/CUDA 组合可能不兼容，可在训练脚本中移除 `attn_implementation=flash_attention_2`。

---

## 10. 验收标准
- 推理：`python tools/inference_sts.py` 全部段落成功运行并在 `/data/output/LM/inference/` 生成 wav
- 训练：Stage-1~4 依次产出独立 `OUTPUT_DIR`，日志无致命错误
- 评估：`evaluate_sts.sh` 正常完成

---

## 11. 版本与更新
- 基础镜像：`shenyunhang/pytorch:24.11-py3_2024-1224`
- 依赖清单：`requirements_ds_gpu.txt`
- 如镜像/依赖升级，请在本文件记录变更并复测
