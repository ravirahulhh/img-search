### 以图搜片 PoC

基于 **视频关键帧 + OpenCLIP 向量 + FAISS PQ 索引** 的本地以图搜片可行性验证工程。

#### 1. 环境准备（venv）

在工程根目录执行：

```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# Windows PowerShell: .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

#### 2. 目录结构

- `data/videos/`：待索引的视频文件（你自行放入；若用 URL 流程则可不使用）。
- `data/frames/`：抽取出的关键帧图片（自动生成，按视频分子目录）。
- `data/index/`：向量索引与辅助文件（FAISS index、元数据 JSONL 等）。
- `data/tmp_downloads/`：URL 流程的临时下载目录（每下载并抽帧后即删原视频，仅临时占用）。
- `src/`
  - `config.py`：集中配置（路径、模型名称、PQ 参数等）。
  - `video_extractor.py`：使用 PySceneDetect 抽取关键帧。
  - `downloader.py`：从 URL 列表下载视频、抽帧并删除原视频（不保留视频文件）。
  - `embedding.py`：加载 OpenCLIP 模型，实现图像向量化。
  - `indexer.py`：构建/保存 FAISS PQ 索引。
  - `search.py`：加载索引并进行相似度检索。
  - `cli.py`：命令行入口。

#### 3. 基本使用流程

1. **准备视频**

   将若干视频文件放入：

   - `data/videos/`

2. **抽取关键帧**

   ```bash
   python -m src.cli extract-frames \
     --video-dir data/videos \
     --frames-dir data/frames \
     --metadata-path data/frames_meta.jsonl
   ```

3. **构建向量索引（PQ + FAISS）**

   ```bash
   python -m src.cli build-index \
     --frames-metadata data/frames_meta.jsonl \
     --index-dir data/index
   ```

4. **以图搜片**

   ```bash
   python -m src.cli search \
     --image path/to/query.jpg \
     --index-dir data/index \
     --top-k 10
   ```

命令会在终端输出最相似的关键帧及其来源视频和时间戳。

#### 4. 从 URL 列表采集（不保留原视频）

若本地空间不足以存放大量视频，可只保留关键帧与索引：从 URL 列表逐个下载视频、抽关键帧后立即删除视频文件。**需在系统安装 [yt-dlp](https://github.com/yt-dlp/yt-dlp)**（用于从各 URL 下载视频，支持 YouTube 等常见站点）。

1. **准备 URL 列表文件**（每行一个视频 URL，支持 `#` 注释和空行）：

   ```text
   # 示例 urls.txt
   https://example.com/video1.mp4
   https://example.com/video2.mp4
   ```

2. **执行 `index-from-urls`**（下载 → 抽帧 → 删视频 → 建索引）：

   ```bash
   python -m src.cli index-from-urls \
     --url-list urls.txt \
     --frames-dir data/frames \
     --metadata-path data/frames_meta.jsonl \
     --index-dir data/index
   ```

   若只想先下载并抽帧、稍后再建索引，可加 `--no-build-index`，之后单独执行 `build-index`。

   元数据中的 `video_path` 会保存为来源 URL，便于检索结果溯源。

#### 5. 备注

- 当前工程用于 PoC，默认参数偏保守，可在 `config.py` 中调整：\n
  - 关键帧抽取策略（每场景采样几张）。\n
  - PQ 参数（`nlist`、`m`、`nbits`）及搜索参数（`nprobe`）。\n
- 如需 GPU 加速，请确保本地 PyTorch 安装了带 CUDA 的版本，并在代码中选择合适的设备。

