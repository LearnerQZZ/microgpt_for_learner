# 教学版GPT实现 (microGPT)

这是一个为教学目的设计的GPT（生成式预训练Transformer）实现，包含详细的代码注释和说明，帮助初学者理解GPT的核心原理。
(代码结合说明文档可见CSDN论坛："https://blog.csdn.net/2501_92511385/article/details/158182038?spm=1001.2014.3001.5502")

## 项目简介

本项目是一个简化的、纯Python实现的GPT模型，使用字符级分词，以英文名字数据集为例，展示了GPT的基本工作原理和实现方法。

主要功能：
- 实现了完整的自动微分系统
- 实现了Transformer架构的核心组件
- 支持模型训练和文本生成
- 提供了详细的教程文档

## 项目结构

```
microgpt-open-source/
├── README.md          # 项目说明文档
├── input.txt          # 训练数据集（英文名字列表）
├── microgpt_teaching.py  # 教学版GPT实现代码
└── microgpt_tutorial.md  # 详细的教程文档
```

## 文件说明

- **README.md**：项目的主要说明文档，包含项目简介、结构、使用方法等信息。
- **input.txt**：包含约32000个英文名字的数据集，用于训练GPT模型。
- **microgpt_teaching.py**：教学版GPT的完整实现，包含自动微分系统、Transformer架构、训练和推理过程。
- **microgpt_tutorial.md**：详细的教程文档，解释了GPT的核心原理、代码结构和运行结果。

## 安装和使用

### 环境要求

- Python 3.7+
- 无需安装任何第三方库，纯Python实现

### 运行方法

1. 克隆或下载本项目到本地
2. 进入项目目录
3. 运行以下命令启动训练和推理：

```bash
python microgpt_teaching.py
```

运行后，你将看到模型训练过程和生成的英文名字示例。

## 技术原理

本实现包含了GPT的所有关键组件：

1. **自动微分系统**：用于计算模型参数的梯度
2. **词嵌入和位置编码**：将token和位置信息转换为向量表示
3. **多头注意力机制**：允许模型关注不同位置的信息
4. **前馈神经网络**：进一步处理注意力机制的输出
5. **层归一化**：稳定训练过程
6. **残差连接**：缓解梯度消失问题
7. **Adam优化器**：用于更新模型参数

## 模型超参数

- `n_layer`：Transformer网络的深度（层数），默认值为1
- `n_embd`：网络的宽度（嵌入维度），默认值为16
- `block_size`：注意力窗口的最大上下文长度，默认值为16
- `n_head`：注意力头的数量，默认值为4
- `learning_rate`：学习率，默认值为0.01
- `num_steps`：训练步数，默认值为1000

## 训练结果

模型训练1000步后，损失会降低到约2.65左右，能够生成看起来合理的英文名字，例如：

```
样本  1: kamon
样本  2: ann
样本  3: karai
样本  4: jaire
样本  5: vialan
样本  6: karia
样本  7: yeran
样本  8: anna
样本  9: areli
样本 10: kaina
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎对本项目做出贡献！如果你有任何改进建议或发现了bug，请：

1. Fork本仓库
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request

## 学习资源

### 论文
- **Attention Is All You Need**：Transformer的原始论文
- **Improving Language Understanding by Generative Pre-Training**：GPT-1的论文
- **Language Models are Unsupervised Multitask Learners**：GPT-2的论文
- **Language Models are Few-Shot Learners**：GPT-3的论文

### 教程
- **Neural Networks: Zero to Hero**：Andrej Karpathy的教程，包含GPT的实现
- **The Annotated Transformer**：哈佛大学的Transformer教程
- **Hugging Face Transformers Documentation**：详细的Transformer库文档

### 代码库
- **Hugging Face Transformers**：最流行的Transformer库
- **OpenAI GPT-3**：OpenAI的GPT-3实现

## 致谢

本项目基于Andrej Karpathy的microGPT代码，进行了教学目的的修改和扩展。感谢Andrej Karpathy的精彩教程和开源贡献。

## 联系方式

如果你有任何问题或建议，请通过以下方式联系我：

- GitHub: [LearnerQZZ]
- 电子邮件: [19806331346@163.com]

---


希望本项目能够帮助你理解GPT的核心原理和实现方法！祝你学习愉快！
