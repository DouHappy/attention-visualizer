# Attention 分布可视化

该项目提供一个基于 FastAPI 与 Plotly 的网页，用于浏览存储在 `.npz` 文件中的注意力矩阵。每个注意力矩阵可通过滑条或数字输入框选择样本、层（layer）和头（head）进行查看，并在鼠标悬停时高亮相应的 token 以及注意力分数。

## 功能概览

- 从指定目录加载包含字段 `attentions`、`sources`、`predictions`、`start_id`、`end_id` 的 `.npz` 文件。
- 根据 `start_id`/`end_id` 定位样本，并展示对应层与注意力头的注意力矩阵。
- 使用模型 tokenizer 的 `apply_chat_template` 生成对话 token，作为热力图的横纵坐标。
- 鼠标悬停在热力图时，高亮显示对应的 token 与注意力分数。

## 安装依赖

```bash
pip install -e .
```

## 启动服务

```bash
attention-viewer run \
  --data-dir /path/to/npz/files \
  --tokenizer <huggingface-tokenizer-name>
```

如需加载带有自定义代码的 tokenizer，可添加 `--trust-remote-code` 选项。

启动后访问 `http://localhost:8000` 即可使用可视化界面。
