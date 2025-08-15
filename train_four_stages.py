#!/usr/bin/env python3
"""
VITA-Audio 四阶段训练脚本
自动化完成所有四个训练阶段，无需手动干预
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import json
from datetime import datetime

class VITAAudioTrainer:
    def __init__(self, base_model_path="/data/models/VITA-MLLM/VITA-Audio-Plus-Vanilla"):
        self.base_model_path = base_model_path
        self.project_root = Path("/home/linux/VITA/VITA-Audio")
        self.output_root = Path("/data/output/LM")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 训练配置
        self.config = {
            "seq_length": 32768,
            "learning_rate": 1e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "save_steps": 500,
            "eval_steps": 500,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "bf16": False,  # RTX 2080 Ti 使用 fp16
            "fp16": True,
            "dataloader_num_workers": 4,
        }
        
        # 四个阶段的配置
        self.stages = {
            1: {
                "name": "Audio-Text Alignment",
                "description": "音频-文本对齐训练",
                "config_file": "configs/sts_finetune_stage1.yaml",
                "input_model": self.base_model_path,
                "output_dir": self.output_root / f"stage1_{self.timestamp}",
                "epochs": 2,
            },
            2: {
                "name": "Single MCTP Module Training", 
                "description": "单MCTP模块训练",
                "config_file": "configs/sts_finetune_stage2_demo.yaml",
                "input_model": None,  # 将在运行时设置为stage1的输出
                "output_dir": self.output_root / f"stage2_{self.timestamp}",
                "epochs": 3,
            },
            3: {
                "name": "Multiple MCTP Modules Training",
                "description": "多MCTP模块训练", 
                "config_file": "configs/sts_finetune_stage3_demo.yaml",
                "input_model": None,  # 将在运行时设置为stage2的输出
                "output_dir": self.output_root / f"stage3_{self.timestamp}",
                "epochs": 3,
            },
            4: {
                "name": "Supervised Fine-tuning",
                "description": "监督微调",
                "config_file": "configs/sts_finetune_stage4_demo.yaml", 
                "input_model": None,  # 将在运行时设置为stage3的输出
                "output_dir": self.output_root / f"stage4_{self.timestamp}",
                "epochs": 2,
            }
        }

    def setup_environment(self):
        """设置训练环境"""
        print("🔧 设置训练环境...")
        
        # 设置环境变量
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
            "CUDA_LAUNCH_BLOCKING": "1",
            "PYTHONPATH": f"{self.project_root}/third_party/GLM-4-Voice:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": "0",
            # 禁用用户级 site-packages，避免混入 /home/linux/.local 下的 Python 3.10 包
            "PYTHONNOUSERSITE": "1",
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"   设置 {key}={value}")
        
        # 切换到项目目录
        os.chdir(self.project_root)
        print(f"   工作目录: {os.getcwd()}")

    def create_deepspeed_config(self):
        """创建DeepSpeed配置文件"""
        config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config["learning_rate"],
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config["learning_rate"],
                    "warmup_num_steps": self.config["warmup_steps"]
                }
            },
            "gradient_accumulation_steps": self.config["gradient_accumulation_steps"],
            "gradient_clipping": self.config["max_grad_norm"],
            "steps_per_print": 10,
            "train_batch_size": self.config["batch_size"] * self.config["gradient_accumulation_steps"],
            "train_micro_batch_size_per_gpu": self.config["batch_size"],
            "wall_clock_breakdown": False
        }
        
        config_path = self.project_root / "ds_config_rtx2080ti.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   创建DeepSpeed配置: {config_path}")
        return config_path

    def run_training_stage(self, stage_num):
        """运行指定的训练阶段"""
        stage = self.stages[stage_num]
        print(f"\n🚀 开始Stage {stage_num}: {stage['name']}")
        print(f"   描述: {stage['description']}")
        print(f"   输入模型: {stage['input_model']}")
        print(f"   输出目录: {stage['output_dir']}")
        
        # 创建输出目录
        stage['output_dir'].mkdir(parents=True, exist_ok=True)
        
        # 构建训练命令 - 基于原始shell脚本
        ds_config = self.create_deepspeed_config()
        
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=1",
            "--nnodes=1", 
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "tools/finetune_sts_v4_48_3.py",
            "--log_level", "info",
            "--do_train",
            "--overwrite_output_dir",
            "--tokenizer_name", str(stage['input_model']),
            "--model_name_or_path", str(stage['input_model']),
            "--audio_tokenizer_path", "/data/models/THUDM/glm-4-voice-tokenizer",
            "--audio_tokenizer_type", "sensevoice_glm4voice",
            "--dataset_name", f"configs/sts_finetune_stage{stage_num}_demo.yaml",
            "--bf16", "False",  # RTX 2080 Ti 兼容
            "--fp16", "True",
            "--output_dir", str(stage['output_dir']),
            "--num_train_epochs", str(stage['epochs']),
            "--per_device_train_batch_size", str(self.config['batch_size']),
            "--per_device_eval_batch_size", str(self.config['batch_size']),
            "--gradient_accumulation_steps", str(self.config['gradient_accumulation_steps']),
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", str(self.config['save_steps']),
            "--save_total_limit", "3",
            "--learning_rate", str(self.config['learning_rate']),
            "--weight_decay", "0.01",
            "--warmup_steps", str(self.config['warmup_steps']),
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "10",
            "--model_max_length", str(self.config['seq_length']),
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", str(self.config['dataloader_num_workers']),
            "--report_to", "none",
            "--trust_remote_code", "True",
            "--attn_implementation", "eager",  # RTX 2080 Ti 兼容
            "--deepspeed", str(ds_config),  # 启用DeepSpeed
            "--text-audio-interval-ratio", "1", "10", "4", "10",
            "--reset_attention_mask",
            "--reset_position_ids",
            "--create_attention_mask", "false",
            "--create_attention_mask_2d", "false",
        ]
        
        print(f"   执行命令: {' '.join(cmd)}")
        
        # 运行训练
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=False,
                text=True,
                check=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n✅ Stage {stage_num} 完成！")
            print(f"   用时: {duration/3600:.2f} 小时")
            
            # 设置下一阶段的输入模型
            if stage_num < 4:
                self.stages[stage_num + 1]['input_model'] = str(stage['output_dir'])
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Stage {stage_num} 失败！")
            print(f"   错误码: {e.returncode}")
            return False

    def run_all_stages(self, start_stage=1, end_stage=4):
        """运行所有训练阶段"""
        print("🎯 开始VITA-Audio四阶段训练")
        print(f"   基础模型: {self.base_model_path}")
        print(f"   训练阶段: Stage {start_stage} 到 Stage {end_stage}")
        print(f"   时间戳: {self.timestamp}")
        
        # 设置环境
        self.setup_environment()
        
        total_start_time = time.time()
        
        # 依次运行各阶段
        for stage_num in range(start_stage, end_stage + 1):
            success = self.run_training_stage(stage_num)
            if not success:
                print(f"\n🛑 训练在Stage {stage_num}失败，停止后续阶段")
                return False
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        print(f"\n🎉 四阶段训练全部完成！")
        print(f"   总用时: {total_duration/3600:.2f} 小时")
        print(f"   最终模型: {self.stages[end_stage]['output_dir']}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="VITA-Audio 四阶段训练脚本")
    parser.add_argument("--base_model", type=str, 
                       default="/home/linux/VITA/VITA-Audio/VITA-Audio-Balance",
                       help="基础模型路径")
    parser.add_argument("--start_stage", type=int, default=1,
                       help="开始阶段 (1-4)")
    parser.add_argument("--end_stage", type=int, default=4, 
                       help="结束阶段 (1-4)")
    parser.add_argument("--seq_length", type=int, default=32768,
                       help="序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = VITAAudioTrainer(base_model_path=args.base_model)
    
    # 更新配置
    trainer.config.update({
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })
    
    # 运行训练
    success = trainer.run_all_stages(args.start_stage, args.end_stage)
    
    if success:
        print("\n🎊 恭喜！VITA-Audio训练成功完成！")
        sys.exit(0)
    else:
        print("\n💥 训练失败！请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
