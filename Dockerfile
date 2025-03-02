# base image
FROM python:3.11 AS base

# install packages
FROM base AS packages

# # 删除其他源配置文件
# RUN rm -rf /etc/apt/sources.list.d/*
# # 复制自定义镜像源文件
# COPY ./docker/sources.list /etc/apt/sources.list
# if you located in China, you can use aliyun mirror to speed up
RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources

RUN apt-get clean \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get dist-upgrade -y 

RUN apt-get install -y --no-install-recommends gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev

COPY requirements.txt /requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/pkg -r requirements.txt

COPY ./lightrag/api/requirements.txt /api/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/pkg -r /api/requirements.txt

# production stage
FROM base AS production

ENV DEBUG false

ENV EDITION SELF_HOSTED
ENV DEPLOY_ENV PRODUCTION

EXPOSE 9621

# set timezone
ENV TZ UTC

WORKDIR /app

# # 删除其他源配置文件
# RUN rm -rf /etc/apt/sources.list.d/*
# # 复制自定义镜像源文件
# COPY ./docker/sources.list /etc/apt/sources.list
# if you located in China, you can use aliyun mirror to speed up
RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources

RUN apt-get clean \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get dist-upgrade -y 

RUN apt-get install -y --no-install-recommends curl wget vim nodejs ffmpeg libgmp-dev libmpfr-dev libmpc-dev \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

COPY --from=packages /pkg /usr/local
COPY ./lightrag  /app/lightrag
COPY ./lightrag/api  /app/api
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG COMMIT_SHA
ENV COMMIT_SHA ${COMMIT_SHA}

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]