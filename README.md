<h2 align="center">
SAM3-DMS: Decoupled Memory Selection for Multi-target Video Segmentation of SAM3
</h2>
<p align="center">
  <strong>Ruiqi Shen</strong><sup style="font-size: 0.7em;">1</sup>
  ·
  <a href="https://scholar.google.com/citations?user=XlQP0GIAAAAJ&hl=zh-CN"><strong>Chang Liu</strong></a><sup style="font-size: 0.7em;">2✉️</sup>
  ·
  <a href="https://henghuiding.com/"><strong>Henghui Ding</strong></a><sup style="font-size: 0.7em;">1✉️</sup>
</p>

<p align="center">
  <sup style="font-size: 0.7em;">1</sup>Fudan University &nbsp;&nbsp;
  <sup style="font-size: 0.7em;">2</sup>Shanghai University of Finance and Economics
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.09699"><img src="https://img.shields.io/badge/arXiv-2601.09699-b31b1b.svg" alt="arXiv"></a>
</p>

https://github.com/user-attachments/assets/a7f13819-c6f8-425e-9502-b2c7b6d97fc5

![Architecture](demo/fig2.jpg?raw=true)

<em>
<strong>TL;DR:</strong> Built upon SAM3, we focus on simultaneous multi-target video segmentation and propose a training-free decoupled memory selection strategy that shifts SAM3's group-level averaging to individual self-assessment, mitigating memory pollution and identity drift in complex scenarios.
</em>

## ⚙️ Installation

```bash
# create new conda environment
conda create -n sam3_decoupled python=3.12
conda deactivate
conda activate sam3_decoupled

# for pytorch/cuda dependencies
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# clone the repo & install packages
git clone https://github.com/FudanCVL/SAM3_decoupled.git
cd SAM3_decoupled
pip install -e .
```

## 📥 Getting checkpoints
⚠️ Please request access to the checkpoints on the SAM3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

## 🚀 Training and Inference
We follow the same training and inference pipeline as SAM3. For detailed instructions, please see [Evaluation](https://github.com/facebookresearch/sam3/tree/main/sam3/eval), and [Training](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md).


## 🧪 Demo
We provide additional streamlined script for interactive PCS. You can simply specify a video input (mp4 or jpg folder) and enter text prompts via the command line to generate results.

```bash
python interactive_demo.py
Enter video path: # input the video (either mp4 or jpg folder)
Enter text prompt: # input the prompt
```
