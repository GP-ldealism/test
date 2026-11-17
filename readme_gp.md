# 1.环境
推荐使用conda 创建虚拟环境，再通过pip统一管理相关包

## preparation conda or miniconda
conda create -p [your path]
pip install ...
少啥就安装即可

## 模型
下载链接：
官方：https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
镜像（实际可用）：https://hf-mirror.com/ydshieh/256x256_diffusion_uncond_from_blended_diffusion/resolve/main/256x256_diffusion_uncond.pt
3、本地训练一个也行

## 参考
conda env create -f requirements.yml
conda activate lan_paint
python -m app.ui
