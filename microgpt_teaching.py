"""
教学版GPT实现 - 纯Python依赖-free版本

这是一个为教学目的设计的GPT（生成式预训练Transformer）实现，
包含详细的代码注释和说明，帮助初学者理解GPT的核心原理。

本实现基于Andrej Karpathy的原始microGPT代码，
但添加了更详细的注释、解释和教学说明。

@karpathy (原始代码)
@teaching version (添加教学注释)
"""

# 导入必要的标准库
import os       # 用于文件操作，如检查文件是否存在
import math     # 用于数学运算，如对数、指数
import random   # 用于随机操作，如种子设置、随机选择、高斯分布、随机打乱
random.seed(42) # 设置随机种子，确保结果可重现

# 步骤1: 准备数据集
# 我们将使用一个简单的名字数据集作为示例
if not os.path.exists('input.txt'):
    import urllib.request
    # 从GitHub下载名字数据集
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 读取数据集，去除空行并打乱顺序
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"数据集大小: {len(docs)}")

# 步骤2: 创建分词器
# 我们使用字符级分词，将每个字符映射到一个唯一的整数ID
uchars = sorted(set(''.join(docs))) # 提取数据集中的唯一字符，排序后作为词汇表
BOS = len(uchars) # 定义一个特殊的开始标记(Beginning of Sequence)的ID
vocab_size = len(uchars) + 1 # 词汇表大小 = 唯一字符数 + 1(用于BOS标记)
print(f"词汇表大小: {vocab_size}")
print(f"词汇表: {uchars}")
print(f"BOS标记ID: {BOS}")

# 步骤3: 实现自动微分系统
# 这是一个简单的自动微分系统，用于计算梯度
class Value:
    """
    自动微分的核心类，表示计算图中的一个节点
    每个节点包含：
    - data: 节点的标量值
    - grad: 损失函数对该节点的梯度
    - _children: 该节点的子节点（计算图中的依赖）
    - _local_grads: 该节点对其子节点的局部梯度
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python内存优化

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向传播计算的值
        self.grad = 0                   # 反向传播计算的梯度
        self._children = children       # 计算图中的子节点
        self._local_grads = local_grads # 对每个子节点的局部梯度

    # 基本运算符重载，用于构建计算图
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 加法的局部梯度都是1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 乘法的局部梯度是另一个操作数
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # 幂运算的局部梯度是 other * self.data**(other-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # 对数的局部梯度是 1/self.data
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # 指数的局部梯度是 exp(self.data)
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU的局部梯度是 1 (如果self.data > 0) 否则 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # 其他运算符重载
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        反向传播算法：
        1. 构建计算图的拓扑排序
        2. 从输出节点开始，设置其梯度为1
        3. 按照拓扑排序的逆序，计算每个节点的梯度
        """
        # 构建拓扑排序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # 从输出节点开始，设置梯度为1
        self.grad = 1
        
        # 按照拓扑排序的逆序计算梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# 步骤4: 初始化模型参数
# 模型超参数
n_layer = 1     # Transformer网络的深度（层数）
n_embd = 16     # 网络的宽度（嵌入维度）
block_size = 16 # 注意力窗口的最大上下文长度（注意：最长的名字是15个字符）
n_head = 4      # 注意力头的数量
head_dim = n_embd // n_head # 每个注意力头的维度

# 辅助函数：创建随机初始化的矩阵
def matrix(nout, nin, std=0.08):
    """
    创建一个形状为 (nout, nin) 的矩阵，使用高斯分布初始化
    nout: 输出维度
    nin: 输入维度
    std: 高斯分布的标准差
    """
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 模型参数字典
state_dict = {
    'wte': matrix(vocab_size, n_embd),  # 词嵌入矩阵
    'wpe': matrix(block_size, n_embd),  # 位置嵌入矩阵
    'lm_head': matrix(vocab_size, n_embd)  # 语言模型头
}

# 添加每个层的参数
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # 注意力查询权重
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # 注意力键权重
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # 注意力值权重
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # 注意力输出权重
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP第一层权重
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP第二层权重

# 将所有参数展平为单个列表，方便优化器处理
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"模型参数数量: {len(params)}")

# 步骤5: 定义模型架构
# 我们将实现一个简化版的GPT-2架构

def linear(x, w):
    """
    线性层：计算输入x与权重w的矩阵乘法
    x: 输入向量
    w: 权重矩阵
    返回: 输出向量
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    Softmax函数：将logits转换为概率分布
    logits: 未归一化的概率
    返回: 归一化的概率分布
    """
    # 为了数值稳定性，减去最大值
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    RMSNorm（均方根归一化）：归一化输入向量
    x: 输入向量
    返回: 归一化后的向量
    """
    # 计算均方
    ms = sum(xi * xi for xi in x) / len(x)
    # 计算缩放因子
    scale = (ms + 1e-5) ** -0.5
    # 应用缩放
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """
    GPT模型的前向传播
    token_id: 当前 token 的 ID
    pos_id: 当前 token 的位置 ID
    keys: 存储之前的键向量
    values: 存储之前的值向量
    返回: 下一个 token 的 logits
    """
    # 1. 词嵌入和位置嵌入
    tok_emb = state_dict['wte'][token_id] # 词嵌入
    pos_emb = state_dict['wpe'][pos_id] # 位置嵌入
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 合并词嵌入和位置嵌入
    x = rmsnorm(x) # 归一化

    # 2. 遍历每一层
    for li in range(n_layer):
        # 2.1 多头注意力块
        x_residual = x  # 保存残差连接
        x = rmsnorm(x)  # 归一化
        
        # 计算查询、键、值
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # 保存当前的键和值，用于后续位置的注意力计算
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []  # 注意力输出
        # 遍历每个注意力头
        for h in range(n_head):
            hs = h * head_dim  # 当前头的起始位置
            # 提取当前头的查询、键、值
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 计算注意力分数
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            # 计算注意力权重
            attn_weights = softmax(attn_logits)
            # 计算注意力加权和
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            # 将当前头的输出添加到总输出
            x_attn.extend(head_out)
        
        # 注意力输出投影
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        # 应用残差连接
        x = [a + b for a, b in zip(x, x_residual)]
        
        # 2.2 MLP块
        x_residual = x  # 保存残差连接
        x = rmsnorm(x)  # 归一化
        # 第一层线性变换
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        # 应用ReLU激活函数
        x = [xi.relu() for xi in x]
        # 第二层线性变换
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        # 应用残差连接
        x = [a + b for a, b in zip(x, x_residual)]

    # 3. 语言模型头
    logits = linear(x, state_dict['lm_head'])
    return logits

# 步骤6: 初始化优化器
# Adam优化器的超参数
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
# 一阶动量缓冲区
m = [0.0] * len(params)
# 二阶动量缓冲区
v = [0.0] * len(params)

# 步骤7: 训练模型
print("开始训练模型...")
num_steps = 1000 # 训练步数
for step in range(num_steps):
    # 7.1 准备输入数据
    # 随机选择一个文档
    doc = docs[step % len(docs)]
    # 分词并添加BOS标记
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    # 限制序列长度
    n = min(block_size, len(tokens) - 1)

    # 7.2 前向传播
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # 前向传播计算logits
        logits = gpt(token_id, pos_id, keys, values)
        # 计算概率分布
        probs = softmax(logits)
        # 计算交叉熵损失
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    # 计算平均损失
    loss = (1 / n) * sum(losses)

    # 7.3 反向传播
    loss.backward()

    # 7.4 更新参数
    lr_t = learning_rate * (1 - step / num_steps) # 线性学习率衰减
    for i, p in enumerate(params):
        # Adam优化器更新规则
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # 重置梯度
        p.grad = 0

    # 打印训练进度
    print(f"步骤 {step+1:4d} / {num_steps:4d} | 损失 {loss.data:.4f}", end='\r')

# 步骤8: 推理（生成文本）
print("\n--- 推理阶段（生成新的名字）---")
temperature = 0.5 # 控制生成文本的"创造性"，值越小越保守，值越大越随机
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # 从BOS标记开始
    sample = []  # 存储生成的字符
    for pos_id in range(block_size):
        # 前向传播计算logits
        logits = gpt(token_id, pos_id, keys, values)
        # 应用温度缩放
        probs = softmax([l / temperature for l in logits])
        # 根据概率分布随机选择下一个token
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        # 如果遇到BOS标记，停止生成
        if token_id == BOS:
            break
        # 添加到生成结果
        sample.append(uchars[token_id])
    # 打印生成的名字
    print(f"样本 {sample_idx+1:2d}: {''.join(sample)}")
