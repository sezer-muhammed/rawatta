# SPDX-License-Identifier: Apache-2.0
# rawatta-official: the Official VGGT-Omega PyTorch code, for the baseline column.
# Build context = eval/ (run_official.sh does this):
#   BUILDX_BUILDER=default docker build -t rawatta-official:latest -f eval/official/Dockerfile eval
# The official repository (FAIR Noncommercial Research License) is downloaded at build time at
# the commit the paper used; it is not part of this repository. Keep the image local: do not
# push it to a public registry, and never bake weights into it (mount them at run time).
# pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime, pinned by digest
FROM pytorch/pytorch@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

RUN pip install --no-cache-dir "numpy<2" einops==0.8.1 safetensors==0.5.3 opencv-python-headless==4.11.0.86

# The archive's SHA-256 belongs to this commit: change both together.
ARG VGGT_OMEGA_COMMIT=48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
ADD --checksum=sha256:7ee9d87cb0ab0efc42d9092332d3e0809ef379bafe2c7c293354378a5657793c \
    https://github.com/facebookresearch/vggt-omega/archive/${VGGT_OMEGA_COMMIT}.tar.gz /tmp/vggt-omega.tar.gz
RUN tar xzf /tmp/vggt-omega.tar.gz -C /opt \
 && mv /opt/vggt-omega-${VGGT_OMEGA_COMMIT} /opt/vggt-omega \
 && echo ${VGGT_OMEGA_COMMIT} > /opt/vggt-omega/.rawatta-commit \
 && pip install --no-cache-dir --no-deps /opt/vggt-omega \
 && rm /tmp/vggt-omega.tar.gz

COPY omega_src.py score.py /opt/rawatta/eval/
COPY official/time_official.py /opt/rawatta/eval/official/
ENV RAWATTA_OMEGA_SRC=/opt/vggt-omega PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/rawatta
ENTRYPOINT ["python", "/opt/rawatta/eval/official/time_official.py"]
