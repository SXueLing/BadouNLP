# Week 1

## Env Setup

- 创建conda虚拟环境
    ```
    conda create -n pytorch python=3.9
    ```
- 进入pytorch虚拟环境
	```
    conda activate pytorch
    ```

- 在虚拟环境pytorch中安装pytorch

    ```
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
    ```