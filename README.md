# 错别字纠错 Attention 分布可视化

该项目提供一个基于 FastAPI、Typer 与 Plotly 的网页应用，用于浏览存储在 `.npz` 文件中的注意力矩阵，并将其与模型的输入输出文本对齐。工具可以帮助快速定位模型回答中与输入差异较大的 token，并观察不同层、不同注意力头之间的注意力权重分布。

## 功能概览

- 🔁 **数据浏览**：通过 Sample ID 滑条或数字输入框切换样本。
- 🧭 **分层查看**：支持通过滑条切换注意力矩阵的层（layer）与注意力头（head），滑条与数字输入框同步更新。
- ✏️ **差异高亮**：基于源文本与预测文本的最小编辑距离，自动标记“需要修改”的 token，并在 token 列表与热力图上突出显示首个错误对应的行列交叉点。
- 🧷 **Token 对齐**：通过 tokenizer 的 `apply_chat_template` 与 `align_tokens_to_attention` 对齐策略，确保热力图轴与显示 token 完整一致，自动剔除尾部 padding。
- 🖱️ **交互提示**：鼠标悬停热力图时，界面会同步高亮对应行列 token 并显示精确注意力值，移出后自动恢复。高亮Prediction中第一个错别字token向Source中对应错别字token在热力图中位置。
- 📄 **上下文信息**：界面实时展示当前样本的原始输入（Source）、模型预测（Prediction）以及所属文件的元数据。

## 数据要求

`AttentionDataset` 默认读取目录中扩展名为 `.npz` 的文件，并期望其中包含下列键：

| 键名 | 类型 | 说明 |
| --- | --- | --- |
| `attentions` | `float32` ndarray，形状 `(layer, batch, head, seq, seq)` | 注意力权重矩阵。
| `sources` | 字符串数组 | 每个样本的原始输入。
| `predictions` | 字符串数组 | 每个样本的模型输出。
| `start_id` | 标量整数 | 文件内样本起始 ID（包含）。
| `end_id` | 标量整数 | 文件内样本结束 ID（不包含）。

> **提示**：`get_sample(sample_id)` 会根据 `start_id <= sample_id < end_id` 选择文件，并使用 `sample_id - start_id` 作为 batch 索引提取注意力矩阵及文本。

## 安装

推荐使用虚拟环境，安装项目依赖：

```bash
pip install -e .
```

## 命令行使用

项目提供 `attention-viewer` CLI（由 Typer 实现），常用命令如下：

```bash
attention-viewer run \
  --data-dir /path/to/npz/files \
  --tokenizer <huggingface-tokenizer-name> \
  [--host 0.0.0.0] \
  [--port 8000] \
  [--reload] \
  [--trust-remote-code]
```

- `--data-dir`：存放 `.npz` 注意力文件的目录。
- `--tokenizer`：Hugging Face 上可用的 tokenizer 名称，需与生成注意力矩阵的模型保持一致。
- `--reload`：开发环境下建议开启，以便自动重载。
- `--trust-remote-code`：当 tokenizer 依赖远程自定义代码时需要显式开启。

启动服务后访问 `http://localhost:8000` 即可看到可视化界面。

## 界面速览

1. **顶部控制面板**：通过滑条或数字输入框选择 Sample ID、Layer、Head，三者双向联动。
2. **文本与文件信息**：展示当前样本的 Source、Prediction 以及所属文件名、样本范围和批次索引。
3. **Token 列表**：显示对齐后的 token 序列，错误 token 会以红色（源）、绿色（预测）背景标记。
4. **注意力热力图**：热力图左上角对齐，第一列为自动插入的 Instruction，Plotly 提供缩放、导出等交互能力。
5. **悬停详情**：在热力图上悬停会更新右侧 Token 列表的高亮状态，并在信息栏显示行列索引及注意力分数。

## 常见问题

- **为什么我看不到任何样本？**
  请确认 `.npz` 文件位于 `--data-dir` 指定目录，并且包含上述必需字段。如果目录为空或字段缺失，应用会在启动时抛出错误。

- **如何扩展功能？**
  可在 `attention_viewer/static/app.js` 中调整前端交互逻辑，或在 `attention_viewer/server.py` 中扩展 API。新添加的 `.js` / `.py` 代码无需额外构建步骤，刷新浏览器即可看到效果。

欢迎提交 Issue 或 Pull Request 来改进该工具！
