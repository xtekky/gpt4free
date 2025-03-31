# PydanticAI 与 G4F 客户端集成

本 README 提供了如何将 PydanticAI 与 G4F 客户端集成以创建与语言模型交互的代理的概述。通过此设置，您将能够应用补丁以使用 PydanticAI 模型、启用调试并同步运行简单的基于代理的交互。但是，请注意，AI 请求中的工具调用目前在此环境中**尚未完全支持**。

## 要求

在开始之前，请确保您已安装以下 Python 依赖项：

- `g4f`：一个与各种 LLM 交互的客户端。
- `pydantic_ai`：一个提供与基于 Pydantic 模型集成的模块。

### 安装

要安装这些依赖项，您可以使用 `pip`：

```bash
pip install g4f pydantic_ai
```

## 分步设置

### 1. 补丁 PydanticAI 以使用 G4F 模型

为了使用 PydanticAI 与 G4F 模型，您需要对客户端应用必要的补丁��这可以通过从 `g4f.integration.pydantic_ai` 导入 `patch_infer_model` 来完成。`api_key` 参数是可选的，因此如果您有一个，可以提供它。如果没有，系统将继续进行而不需要它。

```python
from g4f.integration.pydantic_ai import patch_infer_model

patch_infer_model(api_key="your_api_key_here")  # 可选
```

如果您没有 API 密钥，只需省略 `api_key` 参数。

### 2. 启用调试日志记录

为了进行故障排除和监控，您可能需要启用调试日志记录。这可以通过将 `g4f.debug.logging` 设置为 `True` 来实现。

```python
import g4f.debug

g4f.debug.logging = True
```

这将记录有关内部过程和交互的详细信息。

### 3. 创建一个简单的代理

现在您可以创建一个可以与 LLM 交互的简单代理。代理使用一个模型进行初始化，您还可以定义一个系统提示。以下是一个示例，其中创建了一个带有模型 `g4f:Gemini:Gemini` 和一个简单系统提示的基本代理：

```python
from pydantic_ai import Agent

# 定义代理
agent = Agent(
    'g4f:Gemini:Gemini', # g4f:provider:model_name 或 g4f:model_name
    system_prompt='Be concise, reply with one sentence.',
)
```

### 4. 同步运行代理

设置代理后，您可以同步运行它以与 LLM 交互。`run_sync` 方法将查询发送到 LLM 并返回结果。

```python
# 使用用户查询同步运行代理
result = agent.run_sync('Where does "hello world" come from?')

# 输出响应
print(result.data)
```

在此示例中，代理将系统提示与用户查询（`"Where does 'hello world' come from?"`）一起发送到 LLM。LLM 将处理请求并返回简洁的答案。

### 示例输出

```bash
The phrase "hello world" is commonly used in programming tutorials to demonstrate basic syntax and the concept of outputting text to the screen.
```

## 工具调用和限制

**重要**：工具调用（例如在 AI 请求中应用外部函数或调用 API）**目前尚未完全支持**。如果您的系统依赖于在与模型对话期间调用特定的外部工具或函数，您需要在代理上下文之外实现此功能，或在代理请求之前或之后处理它。

例如，您可以在将数据传递给代理之前处理查询或与外部系统交互。

---

### 不使用 `patch_infer_model` 的简单示例

```python
from pydantic_ai import Agent
from g4f.integration.pydantic_ai import AIModel

agent = Agent(
    AIModel("gpt-4o"),
)

result = agent.run_sync('Are you gpt-4o?')
print(result.data)
```

此示例展示了如何使用特定模型（`gpt-4o`）初始化代理并同步运行它。

---

### 带有工具调用的完整示例：

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from g4f.integration.pydantic_ai import AIModel
from g4f.Provider import PollinationsAI


class MyModel(BaseModel):
    city: str
    country: str

nt = Agent(AIModel(
    "gpt-4o", # 指定提供商和模型
    PollinationsAI # 使用支持的提供商来处理基于工具的响应格式
), result_type=MyModel, model_settings=ModelSettings(temperature=0))

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A.')
    print(result.data)
    print(result.usage())
```

此示例演示了如何使用自定义 Pydantic 模型（`MyModel`）从响应中捕获结构化数据（城市和国家），并使用特定的模型设置运行代理。

---

### 支持不支持工具调用的模型/提供商

对于不完全支持工具调用或缺乏直接 API 以获取结构化输出的模型/提供商，可以使用 `ToolSupportProvider` 来弥补这一差距。此提供商确保代理正确格式化响应，即使模型本身没有内置的结构化输出支持。它通过利用工具列表并在仅使用一个工具时创建响应格式来实现这一点。

### 不支持工具的模型/提供商示例（单工具使用）

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from g4f.integration.pydantic_ai import AIModel
from g4f.providers.tool_support import ToolSupportProvider

from g4f import debug
debug.logging = True

# 定义用于结构化输出的自定义模型（例如，城市和国家）
class MyModel(BaseModel):
    city: str
    country: str

# 为具有工具支持的模型创建代理（使用一个工具）
agent = Agent(AIModel(
    "OpenaiChat:gpt-4o", # 指定提供商和模型
    ToolSupportProvider # 使用 ToolSupportProvider 处理基于工具的响应格式
), result_type=MyModel, model_settings=ModelSettings(temperature=0))

if __name__ == '__main__':
    # 使用查询运行代理以提取信息（例如，城市和国家）
    result = agent.run_sync('European city with the bear.')
    print(result.data)  # 城市和国家的结构化输出
    print(result.usage()) # 使用统计
```

### 解释：

- **`ToolSupportProvider` 作为桥梁**：`ToolSupportProvider` 充当代理和模型之间的桥梁，确保响应格式化为结构化输出，即使模型没有直接支持此类格式的 API。
  
  - 例如，如果模型生成原始文本或非结构化数据，`ToolSupportProvider` 将其转换为预期格式（如 `MyModel`），使代理能够将其处理为结构化数据。
  
- **模型初始化**：我们使用 `PollinationsAI:openai` 模型初始化代理，该模型可能没有内置的 API 来返回结构化输出。相反，它依赖于 `ToolSupportProvider` 来格式化输出。

- **自定义结果模型**：我们定义了一个自定义 Pydantic 模型（`MyModel`）来捕获预期的结构化输出（例如，`city` 和 `country` 字段）。这有助于确保即使模型不支持结构化数据，代理也能解释和格式化它。

- **调试日志记录**：启用 `g4f.debug.logging` 以提供详细的日志记录，用于故障排除和监控代理的执行。

### 示例输出：

```bash
city='Berlin'
country='Germany'
usage={'prompt_tokens': 15, 'completion_tokens': 50}
```

### 关键点：

- **`ToolSupportProvider` 角色**：`ToolSupportProvider` 确保代理将模型的原始或非结构化响应格式化为结构化格式，即使模型本身缺乏内置的结构化数据支持。
  
- **单工具使用**：`ToolSupportProvider` 特别适用于仅使用一个工具的模型，并且需要将模型的输出格式化或转换为结构化响应而无需额外工具。

### 注意事项：

- 此方法适用于返回非结构化文本或需要转换为结构化格式（例如 Pydantic 模型）的数据的模型。
- `ToolSupportProvider` 弥合了模型输出和预期结构化格式之间的差距，使其能够无缝集成到需要结构化响应的工作流中。

---

## LangChain 集成示例

对于使用 LangChain 的用户，以下是一个演示如何将 G4F 模型集成到 LangChain 环境中的示例：

```python
from g4f.integration.langchain import ChatAI
import g4f.debug

# 启用调试日志
g4f.debug.logging = True

llm = ChatAI(
    model="llama3-70b-8192",
    provider="Groq",
    api_key=""  # 可选地在此处添加您的 API 密钥
)

messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4 🦜"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5 🦜"},
    {"role": "user", "content": "3 🦜 4"},
]

response = llm.invoke(messages)
assert(response.content == "7 🦜")
```

此示例展示了如何使用 LangChain 的 `ChatAI` 集成创建一个带有 G4F 模型的对话代理。交互在给定消息的情况下逐步进行，代理处理它们以返回预期的输出。

---

## 结论

通过遵循这些步骤，您已成功将 PydanticAI 模型集成到 G4F 客户端中，创建了一个代理并启用了调试。这使您能够与语言模型进行对话，传递系统提示并同步检索响应。

### 注意事项：
- 调用 `patch_infer_model` 时，`api_key` 参数是可选的。如果您不提供它，系统仍将正常工作而无需 API 密钥。
- 修改代理的 `system_prompt` 以适应您希望进行的对话的性质。
- **AI 请求中的工具调用尚未完全支持**。使用代理的基本功能生成响应，并单独处理外部调用。

有关进一步的自定义和高级用例，请参阅 G4F 和 PydanticAI 文档。
