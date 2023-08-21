# 비용을 줄이는 LLM 학습 기법

- https://fastcampus.co.kr/data_online_prompt

## 설치

### pytorch 설치
- pytorch의 경우 공식 홈페이지의 설치법을 이용해 설치
- colab에는 이미 설치되어있음

### Colab에서

- pytorch는 이미 설치되어 있음
- requirements.txt 파일을 옮기고 아래 커맨드로 설치

```
!pip install -r requirements.txt
```

### Terminal에서

- pytorch는 pytorch 공식 홈페이지에서 설치
  - https://pytorch.org

```
pip install -r requirements.txt
```

## Accelerate Config 설정

- A100의 경우 아래의 설정을 사용하시면 됩니다.
- A100 이상의 GPU만 bf16를 지원하기 때문에 v100, T4 등의 GPU에서는 fp16을 사용하거나 half precision을 사용하지 않고 그대로 fp32를 사용해야 합니다.

```bash
$ accelerate config                                                                     [4:18:27]
[INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
--In which compute environment are you running?
This machine
--Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]:
--What should be your DeepSpeed's ZeRO optimization stage?
2
--Where to offload optimizer states?
cpu
--Where to offload parameters?
cpu
How many gradient accumulation steps you're passing in your script? [1]:
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:2
--Do you wish to use FP16 or BF16 (mixed precision)?
bf16

```
