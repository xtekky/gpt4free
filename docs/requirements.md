### G4F - 额外要求

#### 介绍

您可以部分或完全安装所需的依赖项。因此，G4F 可以根据您的需求进行使用。您有以下选项：

#### 选项

安装 g4f 及其所有可能的依赖项：
```
pip install -U g4f[all]
```
或者仅安装 g4f 和 OpenaiChat 提供商所需的包：
```
pip install -U g4f[openai]
```
安装干扰 API 所需的包：
```
pip install -U g4f[api]
```
安装 Web UI 所需的包：
```
pip install -U g4f[gui]
```
安装上传/生成图像所需的包：
```
pip install -U g4f[image]
```
安装支持 aiohttp 代理的所需包：
```
pip install -U aiohttp_socks
```
安装从浏览器加载 cookies 所需的包：
```
pip install browser_cookie3
```
安装所有包并卸载此包以禁用浏览器支持：
```
pip uninstall nodriver 
```

---
[返回首页](/)
