# TinyLlama Sample

## 项目来源
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

## 模型来源
### model.safetensors下载
1. 运行sh 00_init.sh即可下载到本地data/llm/tinyllama路径下，并设置软连接。也可以用下述方式手动下载
2. 公网下载
```
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/model.safetensors
```

- 下载的model.safetensors请放置到本地data/llm/tinyllama路径下（基于sample根目录）

## 版本支持
目前该模型仅支持V200上执行

## 问答示例
### 问题1：How many people there are in China?
As of 2021, China has a population of approximately 1.4 billion people.
### 问题2：Who are you?
I am a machine learning model trained on a large dataset of text data. I can analyze and understand human language, and I can generate responses based on the context and the user's preferences.
### 问题3：Do you know about China?
Yes, I do know about China. China is a country located in East Asia, bordering Vietnam, Laos, and Cambodia. It is the world's second-largest country by area, after Russia, and the most populous country, with over 1.4 billion people. China has a complex political system with a long history, and it has been a major economic and cultural force in the world for centuries.

## 内存使用情况
需要使用4G PROV200SDK镜像，SMR至少需要设置2G