# JoeyMiniMind

基于 GitHub 开源项目 MiniMind 思路手动实现的轻量级语言模型项目。当前仓库包含一个从零实现的 Causal LM 主体、预训练数据读取、预训练脚本和交互式推理脚本，主要用于学习小型 Transformer / LLM 的核心训练与推理流程。

代码中的模型命名为 `MokioMind`，仓库名保留为 `joeyminimind`。

## 项目特点

- 手动实现 Decoder-only Transformer 结构
- 支持 RMSNorm、RoPE、GQA、KV Cache 和 tied embedding
- 兼容 Hugging Face `PreTrainedModel` / `GenerationMixin` 推理接口
- 提供预训练数据集封装和预训练脚本
- 支持单卡训练，也预留了 DDP 分布式训练逻辑
- 使用本地 `model/` 目录中的 tokenizer 配置

## 目录结构

```text
.
├── dataset/
│   └── lm_dataset.py          # 预训练数据集读取
├── method/
│   └── rmsnrom.py             # RMSNorm 实验/方法文件
├── model/
│   ├── model.py               # MokioMind 模型实现
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── trainer/
│   ├── train_pretrain.py      # 预训练入口
│   └── trainer_utils.py       # 学习率、checkpoint、DDP 等工具
├── eval.py                    # 推理/对话入口
├── main.py                    # uv 默认示例入口
├── pyproject.toml
└── uv.lock
```

## 环境准备

项目使用 Python 3.12，并通过 `uv` 管理依赖。

```bash
uv sync
```

当前远端版本默认使用 CUDA 12.1 对应的 PyTorch 源：

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

如果只在 CPU 环境运行，需要按自己的环境调整 `pyproject.toml` 中的 PyTorch 依赖和源配置。

## 数据准备

预训练脚本默认读取：

```text
dataset/pretrain_t2t_mini.jsonl
```

数据文件没有包含在仓库中，需要自行准备。每行是一个 JSON 对象，并包含 `text` 字段：

```json
{"text": "这里是一段用于语言模型预训练的文本。"}
```

## 预训练

训练脚本里的默认路径按 `trainer/` 目录作为工作目录设计，因此建议从 `trainer` 目录启动：

```bash
cd trainer
uv run python train_pretrain.py
```

常用参数示例：

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

训练产物默认保存到：

```text
out/pretrain_512.pth
checkpoints/pretrain_512.pth
checkpoints/pretrain_512_resume.pth
```

这些权重和 checkpoint 文件已在 `.gitignore` 中排除，不会提交到仓库。

## 推理

训练得到权重后，可以在项目根目录运行：

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

如果使用 `--load_from model`，脚本会加载本项目手动实现的 `MokioMindForCausalLM`，并从 `out/{weight}_{hidden_size}.pth` 读取权重。传入其他 Hugging Face 模型路径时，会走 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`。

## 重要说明

- 仓库不包含训练数据和模型权重，需要自行准备。
- `eval.py` 中保留了 LoRA 相关参数，但当前仓库没有提供 `model_lora` 实现；默认 `--lora_weight None` 即可。
- `main.py` 只是 uv 创建项目时的简单示例入口，核心训练和推理入口分别是 `trainer/train_pretrain.py` 与 `eval.py`。
- 该项目主要用于学习和复现 MiniMind 类小模型的实现流程，不建议直接当作生产级训练框架使用。
