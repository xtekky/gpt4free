# G4F Docker 设置

## 目录
   - [先决条件](#先决条件)
   - [安装和设置](#安装和设置)
   - [测试 API](#测试-api)
   - [故障排除](#故障排除)
   - [停止服务](#停止服务)


## 先决条件
**在开始之前，请确保您的系统上已安装以下内容：**
   - [Docker](https://docs.docker.com/get-docker/)
   - [Docker Compose](https://docs.docker.com/compose/install/)
   - Python 3.7 或更高版本
   - pip (Python 包管理器)

**注意：** 如果您遇到 Docker 问题，可以直接使用 Python 运行项目。

## 安装和设置

### Docker 方法（推荐）
1. **克隆仓库**
   ```bash
   git clone https://github.com/xtekky/gpt4free.git
   cd gpt4free
   ```

2. **使用 Docker Compose 构建和运行**

   拉取最新镜像并运行带有 Google Chrome 支持的容器：
   ```bash
      docker pull hlohaus789/g4f
      docker-compose up -d
   ```
   或运行不带 Google Chrome 的小型 Docker 镜像：
   ```bash
      docker-compose -f docker-compose-slim.yml up -d
   ```

3. **访问 API 或 GUI**

   API 服务器将可通过 `http://localhost:1337` 访问

   GUI 可通过此 URL 访问：`http://localhost:8080`

### 非 Docker 方法
如果您遇到 Docker 问题，可以直接使用 Python 运行项目：

1. **克隆仓库**
   ```bash
   git clone https://github.com/xtekky/gpt4free.git
   cd gpt4free
   ```

2. **安装依赖项**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行服务器**
   ```bash
   python -m g4f.api.run
   ```

4. **访问 API 或 GUI**

   API 服务器将可通过 `http://localhost:1337` 访问

   GUI 可通过此 URL 访问：`http://localhost:8080`


## 测试 API
**您可以使用 curl 或创建一个简单的 Python 脚本来测试 API：**
### 使用 curl
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of France?"}' http://localhost:1337/chat/completions
```

### 使用 Python
**创建一个名为 `test_g4f.py` 的文件，内容如下：**
```python
import requests

url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "gpt-4o-mini", 
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}

json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```

**运行脚本：**
```bash
python test_g4f.py
```

## 故障排除
- 如果您遇到 Docker 问题，请尝试按照非 Docker 方法直接使用 Python 运行项目。
- 确保您有运行 Docker 命令的必要权限。您可能需要使用 `sudo` 或将您的用户添加到 `docker` 组。
- 如果服务器未启动，请检查日志中的任何错误消息，并确保所有依赖项已正确安装。

**_有关 API 端点和使用的详细信息，请参阅 [G4F API 文档](docs/interference-api.md)。_**



## 停止服务

### Docker 方法
**要停止 Docker 容器，请使用以下命令：**
```bash
docker-compose down
```

### 非 Docker 方法
如果您使用 Python 直接运行服务器，可以在运行的终端中按 Ctrl+C 停止它。

---

[返回首页](/)
