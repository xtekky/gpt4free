# G4F - Git 安装指南

本指南提供了使用 Git 从源代码安装 G4F 的分步说明。

## 目录

1. [先决条件](#先决条件)
2. [安装步骤](#安装步骤)
   1. [克隆仓库](#1-克隆仓库)
   2. [导航到项目目录](#2-导航到项目目录)
   3. [设置 Python 虚拟环境](#3-设置-python-虚拟环境-推荐)
   4. [激活虚拟环境](#4-激活虚拟环境)
   5. [安装依赖项](#5-安装依赖项)
   6. [验证安装](#6-验证安装)
3. [使用](#使用)
4. [故障排除](#故障排除)
5. [其他资源](#其他资源)

---

## 先决条件

在开始之前，请确保您的系统上已安装以下内容：
- Git
- Python 3.7 或更高版本
- pip (Python 包管理器)

## 安装步骤

### 1. 克隆仓库
**打开终端并运行以下命令以克隆 G4F 仓库：**
```bash
git clone https://github.com/xtekky/gpt4free.git
```

### 2. 导航到项目目录
**切换到项目目录：**
```bash
cd gpt4free
```

### 3. 设置 Python 虚拟环境 (推荐)
**建议使用虚拟环境来管理项目依赖项：**
```bash
python3 -m venv venv
```

### 4. 激活虚拟环境
**根据您的操作系统激活虚拟环境：**
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

- **macOS 和 Linux:**
  ```bash
  source venv/bin/activate
  ```

### 5. 安装依赖项
**您有两种安装依赖项的选项：**

#### 选项 A: 安装最小依赖项
**对于轻量级安装，使用：**
```bash
pip install -r requirements-min.txt
```

#### 选项 B: 安装所有包
**要进行完整安装并启用所有功能，使用：**
```bash
pip install -r requirements.txt
```

### 6. 验证安装
您现在可以创建 Python 脚本并使用 G4F 功能。以下是一个基本示例：

**在根文件夹中创建一个 `g4f-test.py` 文件并开始使用仓库：**
```python
import g4f
# 在此处编写您的代码
```

## 使用
**安装完成后，您可以在 Python 脚本中开始使用 G4F。以下是一个基本示例：**
```python
import g4f

# 在此处编写您的 G4F 代码
# 例如：
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
    # 添加任何其他必要的参数
)

print(response.choices[0].message.content)
```

## 故障排除
**如果在安装或使用过程中遇到任何问题：**
   1. 确保所有先决条件已正确安装。
   2. 确保您在正确的目录中并且虚拟环境已激活。
   3. 尝试重新安装依赖项。
   4. 查阅 [G4F 文档](https://github.com/xtekky/gpt4free) 以获取更详细的信息。

## 其他资源
   - [G4F GitHub 仓库](https://github.com/xtekky/gpt4free)
   - [Python 虚拟环境指南](https://docs.python.org/3/tutorial/venv.html)
   - [pip 文档](https://pip.pypa.io/en/stable/)

---

**_有关更多信息或支持，请访问 [G4F GitHub 问题页面](https://github.com/xtekky/gpt4free/issues)._**

---  
[返回首页](/)
