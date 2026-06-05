# JoeyMiniMind

JoeyMiniMind is a lightweight language-model learning project manually implemented with reference to the open-source MiniMind project. It contains a from-scratch Causal LM implementation, a pretraining dataset wrapper, a pretraining script, and an interactive inference script.

The model code is named `MokioMind`, while the repository keeps the name `joeyminimind`.

## Highlights

- Manually implemented Decoder-only Transformer
- RMSNorm, RoPE, GQA, KV Cache, and tied token embeddings
- Hugging Face-compatible `PreTrainedModel` and `GenerationMixin` interfaces
- Local tokenizer files under `model/`
- Pretraining dataset loader for JSON/JSONL text data
- Single-GPU training by default, with DDP logic prepared in the trainer
- Simple interactive inference script for local model weights

## Project Structure

```text
.
|-- dataset/
|   `-- lm_dataset.py          # Pretraining dataset loader
|-- method/
|   `-- rmsnrom.py             # RMSNorm experiment/helper file
|-- model/
|   |-- model.py               # MokioMind model implementation
|   |-- tokenizer.json
|   `-- tokenizer_config.json
|-- trainer/
|   |-- train_pretrain.py      # Pretraining entry point
|   `-- trainer_utils.py       # LR schedule, checkpoint, DDP helpers
|-- eval.py                    # Inference/chat entry point
|-- main.py                    # Default uv sample entry point
|-- pyproject.toml
`-- uv.lock
```

## Requirements

- Python 3.12
- `uv`
- CUDA-compatible PyTorch is configured by default in the current remote version

Install dependencies:

```bash
uv sync
```

The current dependency configuration uses the CUDA 12.1 PyTorch index:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

For CPU-only environments, adjust the PyTorch dependency and index settings in `pyproject.toml` for your machine.

## Data Format

The pretraining script expects a JSON/JSONL file with a `text` field. The default path is:

```text
dataset/pretrain_t2t_mini.jsonl
```

Example line:

```json
{"text": "This is a piece of text used for language model pretraining."}
```

Training data is not included in this repository.

## Pretraining

The pretraining script uses paths that are designed around `trainer/` as the working directory. Start it from that folder:

```bash
cd trainer
uv run python train_pretrain.py
```

Example:

```bash
uv run python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 32 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 512
```

Default output paths:

```text
out/pretrain_512.pth
checkpoints/pretrain_512.pth
checkpoints/pretrain_512_resume.pth
```

Model weights, checkpoints, datasets, and runtime outputs are ignored by Git.

## Inference

After training a local checkpoint, run from the repository root:

```bash
uv run python eval.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
  --hidden_size 512 \
  --num_hidden_layers 8
```

Runtime prompt:

```text
[0] auto test
[1] manual input
```

When `--load_from model` is used, `eval.py` constructs `MokioMindForCausalLM` from this repository and loads weights from:

```text
out/{weight}_{hidden_size}.pth
```

If another Hugging Face model path is passed to `--load_from`, the script uses `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

## Notes

- This repository does not include training data or model weights.
- `eval.py` still exposes LoRA-related arguments, but this repository currently does not include a `model_lora` implementation. Keep `--lora_weight None` unless you add that implementation yourself.
- `main.py` is only the default `uv` sample entry point. The main training and inference entry points are `trainer/train_pretrain.py` and `eval.py`.
- This project is intended for learning and reproducing the implementation flow of small MiniMind-style language models. It is not a production training framework.

---

# JoeyMiniMind 中文说明

JoeyMiniMind 是一个参考 GitHub 开源项目 MiniMind 思路、手动实现的轻量级语言模型学习项目。仓库包含从零实现的 Causal LM 主体、预训练数据读取、预训练脚本和交互式推理脚本，主要用于理解小型 Transformer / LLM 的核心训练与推理流程。

代码中的模型命名为 `MokioMind`，仓库名保留为 `joeyminimind`。

## 项目特点

- 手动实现 Decoder-only Transformer 结构
- 支持 RMSNorm、RoPE、GQA、KV Cache 和 tied embedding
- 兼容 Hugging Face `PreTrainedModel` / `GenerationMixin` 推理接口
- 使用本地 `model/` 目录中的 tokenizer 配置
- 提供 JSON/JSONL 文本数据的预训练数据集封装
- 默认支持单卡训练，也预留了 DDP 分布式训练逻辑
- 提供简单的本地权重交互式推理脚本

## 目录结构

```text
.
|-- dataset/
|   `-- lm_dataset.py          # 预训练数据集读取
|-- method/
|   `-- rmsnrom.py             # RMSNorm 实验/辅助文件
|-- model/
|   |-- model.py               # MokioMind 模型实现
|   |-- tokenizer.json
|   `-- tokenizer_config.json
|-- trainer/
|   |-- train_pretrain.py      # 预训练入口
|   `-- trainer_utils.py       # 学习率、checkpoint、DDP 等工具
|-- eval.py                    # 推理/对话入口
|-- main.py                    # uv 默认示例入口
|-- pyproject.toml
`-- uv.lock
```

## 环境准备

- Python 3.12
- `uv`
- 当前远端版本默认配置 CUDA 版 PyTorch

安装依赖：

```bash
uv sync
```

当前依赖配置使用 CUDA 12.1 对应的 PyTorch 源：

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

如果只在 CPU 环境运行，需要按自己的机器环境调整 `pyproject.toml` 中的 PyTorch 依赖和源配置。

## 数据格式

预训练脚本读取包含 `text` 字段的 JSON/JSONL 文件。默认路径是：

```text
dataset/pretrain_t2t_mini.jsonl
```

示例：

```json
{"text": "这里是一段用于语言模型预训练的文本。"}
```

训练数据不包含在本仓库中，需要自行准备。

## 预训练

训练脚本中的默认路径按 `trainer/` 目录作为工作目录设计，因此建议从 `trainer` 目录启动：

```bash
cd trainer
uv run python train_pretrain.py
```

示例：

```bash
uv run python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight pretrain \
  --epochs 1 \
  --batch_size 32 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 512
```

默认输出路径：

```text
out/pretrain_512.pth
checkpoints/pretrain_512.pth
checkpoints/pretrain_512_resume.pth
```

模型权重、checkpoint、数据集和运行输出都已被 Git 忽略。

## 推理

训练得到本地权重后，在项目根目录运行：

```bash
uv run python eval.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
  --hidden_size 512 \
  --num_hidden_layers 8
```

启动后选择：

```text
[0] 自动测试
[1] 手动输入
```

使用 `--load_from model` 时，`eval.py` 会构建本仓库手动实现的 `MokioMindForCausalLM`，并从以下路径读取权重：

```text
out/{weight}_{hidden_size}.pth
```

如果给 `--load_from` 传入其他 Hugging Face 模型路径，脚本会走 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`。

## 重要说明

- 仓库不包含训练数据和模型权重，需要自行准备。
- `eval.py` 中保留了 LoRA 相关参数，但当前仓库没有提供 `model_lora` 实现；除非自行补充 LoRA 实现，否则保持 `--lora_weight None`。
- `main.py` 只是 `uv` 创建项目时的默认示例入口，核心训练和推理入口分别是 `trainer/train_pretrain.py` 与 `eval.py`。
- 该项目主要用于学习和复现 MiniMind 类小模型的实现流程，不建议直接当作生产级训练框架使用。
