# Comfyui_Qwen3-VL-Instruct-Plus

这是基于[ComfyUI](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct)实现的[Qwen3-VL-Instruct](https://github.com/QwenLM/Qwen3-VL)插件，支持文本查询、视频查询、单图查询和多图查询等功能，可用于生成描述或响应内容。


## 基本工作流

- **文本查询**：用户可以提交文本查询以获取信息或生成描述。例如，用户输入"生命的意义是什么？"即可得到相关回答。

- **视频查询**：当用户上传视频时，系统可以分析内容并为每一帧生成详细描述，或对整个视频进行总结。例如，输入"为给定视频生成描述"即可得到视频相关内容。

- **单图查询**：该工作流支持为单张图片生成描述。用户可以上传一张照片并询问"这张图片展示了什么？"，可能得到类似"一群威严的狮子在草原上休息"的回答。

- **多图查询**：对于多张图片，系统可以提供整体描述或串联图片内容的叙事。例如，输入"根据以下系列图片创作一个故事：一张情侣在海滩的照片、一张婚礼仪式的照片、最后一张是婴儿洗礼的照片"。

> [!IMPORTANT]
> 使用工作流的重要说明
> - 请确保您的ComfyUI环境中已安装"显示文本节点（Display Text node）"。如果遇到该节点缺失的问题，您可以在[ComfyUI_MiniCPM-V-4_5仓库](https://github.com/IuvenisSapiens/ComfyUI_MiniCPM-V-4_5)中找到它。安装这个额外的插件后，"显示文本节点"将可用。


## 安装方法

- 通过[ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)安装（搜索`Qwen3`）

- 下载或通过git克隆本仓库到`ComfyUI\custom_nodes\`目录，然后运行：

```python
pip install -r requirements.txt
```


## 模型下载

所有模型在运行工作流时会自动下载（如果未在`ComfyUI\models\prompt_generator\`目录中找到）。


## 最新功能优化

- 新增模型资源自动清理机制，在切换模型或结束运行时会自动释放显存，减少内存占用
- 优化模型加载逻辑，支持4bit/8bit量化加载，适配不同硬件配置
- 增强多路径输入节点（MultiplePathsInput）功能，可动态调整输入路径数量，提升多图/多视频处理灵活性
- 完善节点交互逻辑，优化工作流中节点连接的稳定性和显示效果


## 许可证

本项目基于Apache License 2.0许可证开源，详情参见[LICENSE](LICENSE)文件。
```
