# An Open-Source DeepSeek API Router

<p align="center">
   <a href="https://github.com/llm-router/DeepSeekRouter/blob/main/README.md">English</a> <a href="https://github.com/llm-router/DeepSeekRouter/blob/main/README-zh.md">中文</a>
</p>

An open-source router for public DeepSeek API services with custom policies (e.g., lowest price, highest output token rate).

Currently, it supports DeepSeek R1 and the following providers:
- [Volcengine Ark](https://www.volcengine.com/)
- [Together.ai](https://www.together.ai/)
- [NVIDIA NIM](https://www.nvidia.com/en-us/ai/)
- [DeepSeek](https://www.deepseek.com/)
- [Azure AI Foundry](https://ai.azure.com/)
- [Fireworks.ai](https://fireworks.ai/)
- [Alibaba Cloud Bailian](https://www.aliyun.com/product/bailian)
- [Tencent Cloud TI](https://cloud.tencent.com/product/ti)
- [Huawei Cloud ModelArts](https://www.huaweicloud.com/product/modelarts/studio.html)
- [SiliconFlow](https://siliconflow.cn/)
- [China Mobile Cloud](https://ecloud.10086.cn/portal)
- [State Cloud](https://www.ctyun.cn/act/xirang/deepseek)

The following table explains the metrics that this tool can evaluate:

| **Abbreviation**         | **Full Form**                                      | **Description**                                                                       | **Supported Statistics**   | **Reference**|
|---------------------|----------------------------------------------------|---------------------------------------------------------------------------------------|----------------------------|----------------------------|
| **ttft**            | Time to first token                                | Latency between input submission and generation of the first output token             | average, median, p90, p99  |
| **prefill**         | Prefill tokens per second                          | # of input tokens / ttft                                                              | average, median, p90, p99  |
| **decode**          | Decode tokens per second                           | # of output tokens / (total time - ttft)                                              | average, median, p90, p99  |
| **OTPS**            | Output tokens per second                           | # of output tokens / total time                                                       | average, median, p90, p99  |
| **OTPR**            | Output tokens per request                          | Total output tokens (including reasoning tokens) generated to answer one user request | average, median, p90, p99  |
| **Success Rate**    | Request success rate                               | Percentage of requests completed without errors/timeouts (regardless of correctness)  | |
| [**GPQA-diamond**](https://github.com/idavidrein/gpqa)           | Subset of Graduate-Level Google-Proof Q&A                    | Benchmark testing **expert-level knowledge** across STEM/humanities                   | |
| [**AIME2024**](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqGg01uE0oKFIeZ8GqperV-fbdCygQgT5_j1yFah7MOzl6C03Ll)       | American Invitational Mathematics Examination 2024 | **Math problem-solving** benchmark based on real competition questions                | |
| [**LiveCodeBench**](https://github.com/LiveCodeBench/LiveCodeBench)   | Live Coding Benchmark                              | Real-time **code generation** evaluation with execution testing                       | |
| [**MMLU**](https://github.com/hendrycks/test)         | Massive Multitask Language Understanding           | 57-subject **multiple-choice test** spanning STEM/humanities                          | |
| [**C-Eval**](https://github.com/hkust-nlp/ceval) | Chinese Evaluation                                 | **Chinese-language** benchmark for STEM/humanities knowledge                          | |
| **Output Price**    | Output token price                                 | CNY per million output tokens                                                         | |
| **Input Price**     | Input token price                                  | CNY per million input tokens                                                          | |

| **Abbreviation**  | **Full Form**                                      | **Description**                                                                       | **Supported Statistics**   | 
|-------------------|----------------------------------------------------|---------------------------------------------------------------------------------------|----------------------------|
| **ttft**          | Time to first token                                | Latency between input submission and generation of the first output token             | average, median, p90, p99  |
| **prefill**       | Prefill tokens per second                          | # of input tokens / ttft                                                              | average, median, p90, p99  |
| **decode**        | Decode tokens per second                           | # of output tokens / (total time - ttft)                                              | average, median, p90, p99  |
| **otps**          | Output tokens per second                           | # of output tokens / total time                                                       | average, median, p90, p99  |
| **otpr**          | Output tokens per request                          | Total output tokens (including reasoning tokens) generated to answer one user request | average, median, p90, p99  |
| **success**       | Request success rate                               | Percentage of requests completed without errors/timeouts (regardless of correctness)  | |
| [**gpqa**](https://github.com/idavidrein/gpqa)          | Graduate-Level Google-Proof Q&A                    | Benchmark testing **expert-level knowledge** across STEM/humanities                   | |
| [**aime2024**](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqGg01uE0oKFIeZ8GqperV-fbdCygQgT5_j1yFah7MOzl6C03Ll)      | American Invitational Mathematics Examination 2024 | **Math problem-solving** benchmark based on real competition questions                | |
| [**livecodebench**](https://github.com/LiveCodeBench/LiveCodeBench) | Live Coding Benchmark                              | **Code generation** evaluation with execution testing                                 | |
| [**mmlu**](https://github.com/hendrycks/test)          | Massive Multitask Language Understanding           | 57-subject **multiple-choice test** spanning STEM/humanities                          | |
| [**ceval**](https://github.com/hkust-nlp/ceval)         | Chinese Evaluation                                 | **Chinese-language** benchmark                          | |
| **output_price** | Output token price                                 | CNY per million output tokens                                                         | |
| **input_price**   | Input token price                                  | CNY per million input tokens                                                          | |

## Install

```bash
conda create -n deepseek_router python=3.10
conda activate deepseek_router
git clone --recurse-submodules git@github.com:llm-router/DeepSeekRouter.git
cd DeepSeekRouter
pip install -e .
```

## Configuration

1. Change the `log` field in `config/cron.json` to where you want to store the logs.
2. Set the `default` field of each provider to your own API key in `config/keys.json`. Ignore those providers of which you do not have a key and those that you do not plan to evaluate.
```
{
  "volcengine": {
    "default": "your-key-for-volcengine"
  },
  "together": {
    "default": "your-key-for-together.ai"
  },
  ...
```
3. You do not need to configure the endpoints of the inference providers except for Azure AI Foundry. You need to change the `base_url` field to your own endpoint.

## [Optional] If you want to use a SQL database to store the results and deploy the dashboard

1. Change the connection string in `config/db.json` to yours. If you use MySQL, it should look like:
```
{
  "conn_str": "mysql+mysqlconnector://user:12345678@localhost:3306/deepseek_router"
}
```
2. Run the following command to create tables:
```bash
python deepseek_router/model/model.py
```
3. Run the following command to start the web app:
```bash
python deepseek_router/app/app.py
```
4. The dashboard is available at `http://localhost:5000`. You need to run the evaluation to see the results. Check out `deepseek_router/app/templates/home.html` and make your changes to the dashboard.

## Evaluate intelligence

You should use `deepseek_router/cron/benchmark_test_all.py` to evaluate intelligence. It can take one day or more to run a dataset.

1. Download the datasets:
```bash
./scripts/download_datasets.sh # Download dataset for opencompass evaluate
```

2. Use `llmqos/cron/benchmark_test_all.py` to evaluate intelligence. It can take one day or more to run a dataset. Use `--output` flag to specify the output format.

Available datasets are under `data/datasets.json`. Note that only `gpqa`, `aime2024` and `ceval` have been tested.

```bash
python deepseek_router/cron/benchmark_test_all.py --dataset <dataset> --output <db OR csv OR both> # run dataset for all providers
```

## Evaluate performance

You should use `deepseek_router/cron/service_perf_test_all.py` to evaluate performance. It can take a few minute for one run. You should run it for enough time in order to get an accurate result.
```bash
python deepseek_router/cron/service_perf_test_all.py --output <db OR csv OR both>
```
By default, we use `ShareGPT` and `L-Eval` datasets for performance testing. You can write your own prompt generator and use that for performance testing.

## Choose the best providers for you

After you have done evaluation, you can provide your requirement and find out the right providers for you. A requirement file looks like:
```
{
  "filters": [
    {
      "metric": "success",
      "min": 0.95
    },
    {
      "metric": "otps",
      "stat": "p90",
      "min": 30
    },
    {
      "metric": "gpqa",
      "min": 60
    }
  ],
  "order": [
    {
      "metric": "output_price"
    }
  ]
}
```

Run the router program:
```Bash
python deepseek_router/router/router.py --requirement <PATH_TO_YOUR_REQUIREMENT_FILE>
```

## Acknowledgments
We have referred to the design of the following projects/websites:
- [Artifical Analysis](https://artificialanalysis.ai/)
- [DeepSeek API Arena](https://deepseek.ai-infra.fun/)

We reused [OpenCompass](https://github.com/open-compass/opencompass) for running benchmarks.



