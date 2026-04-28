FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_x86:9.0.0-beta.1-910b-py3.11
# FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_arm:9.0.0-beta.1-910b-py3.11


RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN pip3 install esdk-obs-python --trusted-host mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple

RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torchtitan==0.2.2 \
    torch==2.10.0+cpu \
    torch_npu==2.10.0rc2 \
    pybind11 \
    triton-ascend==3.2.0 \
    scipy \
    safetensors==0.7.0 \
    pytest==7.3.2 \
    pytest-cov \
    pre-commit \
    pyrefly==0.45.1 \
    transformers \
    einops \
    expecttest \
    tomli_w

COPY ./cluster_smoke_task.sh /home/cluster_smoke_task.sh
COPY ./upload.py /home/upload.py
