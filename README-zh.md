# DeepSeek API Router

<p align="center">
   <a href="https://github.com/llm-router/DeepSeekRouter/blob/main/README.md">English</a> <a href="https://github.com/llm-router/DeepSeekRouter/blob/main/README-zh.md">中文</a>
</p>

本仓库是一个开源的DeepSeek API Router，支持自定义策略(例如最低价格、最高每秒输出token数)来选择合适的API。

目前，它支持DeepSeek R1以及下列API供应商：
- [火山引擎](https://www.volcengine.com/)
- [Together.ai](https://www.together.ai/)
- [NVIDIA NIM](https://www.nvidia.com/en-us/ai/)
- [DeepSeek](https://www.deepseek.com/)
- [Azure AI Foundry](https://ai.azure.com/)
- [Fireworks.ai](https://fireworks.ai/)
- [阿里云百炼](https://www.aliyun.com/product/bailian)
- [腾讯云TI](https://cloud.tencent.com/product/ti)
- [华为云ModelArts](https://www.huaweicloud.com/product/modelarts/studio.html)
- [硅基流动](https://siliconflow.cn/)
- [移动云](https://ecloud.10086.cn/portal)
- [天翼云](https://www.ctyun.cn/act/xirang/deepseek)

## 安装步骤

```bash
conda create -n deepseek_router python=3.10
conda activate deepseek_router
git clone --recurse-submodules git@github.com:llm-router/DeepSeekRouter.git
cd DeepSeekRouter
pip install -e .
```

## 配置

1. 在`config/cron.json`中更改`log`字段，将日志保存到指定位置。
2. 在`config/api_keys.json`中填写API密钥。忽略您没有密钥的API服务商。
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
3. 注意：需要在`data/services.json`中修改`azure`的`base_url`字段，将其替换成自己的API端点。

## [可选] 如果想使用SQL数据库来存储结果并部署网站进行展示

1. 在`config/db.json`中，更改用户名和密码。 下面是使用MySQL的连接字符串示例:
```
{
  "conn_str": "mysql+mysqlconnector://user:12345678@localhost:3306/deepseek_router"
}
```

2. 创建表格:
```bash
python deepseek_router/model/model.py
```

3. 启动网站:
```bash
python deepseek_router/app/app.py
```

4. 网站可在`http://localhost:5000`进行浏览。您需要运行智能水平和系统性能测评后才能看到结果。可以修改`deepseek_router/app/templates/home.html`来定制网站页面。

## 智能水平测评

可以使用 `deepseek_router/cron/benchmark_test_all.py` 来评测性能。注意，单个数据集的测试时间可能在一天以上。

1. 下载数据集:
```bash
./scripts/download_datasets.sh # Download dataset for opencompass evaluate
```

2. 运行`deepseek_router/cron/benchmark_test_all.py`进行测评。请使用`--output`参数来选择保存的格式。

可选的数据集在 `data/datasets.json`中。注意，只有 `gpqa`, `aime2024` 和 `ceval` 已经被完整测试过。

```bash
python deepseek_router/cron/benchmark_test_all.py --dataset <dataset> --output <db OR csv OR both> # run dataset for all providers
```

## 系统性能测评

您可以使用`deepseek_router/cron/service_perf_test_all.py`来评测系统性能。每次实验的测试时间大约是几分钟。为了得到更准确的结果，您可以多次运行。

```bash
python deepseek_router/cron/service_perf_test_all.py --output <db OR csv OR both>
```

`ShareGPT` 和 `L-Eval` 是默认用于系统性能测评的数据集，每次实验随机从中采样。您也可以编写自己的prompt生成器用于测试。

## 选择供应商

在完成测评之后，您可以根据需要筛选出最合适的供应商。你需要提供一个需求文件，它的格式如下:
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

之后，你可以运行下面的代码来查看符合需要的API供应商:
```Bash
python deepseek_router/router/router.py --requirement <PATH_TO_YOUR_REQUIREMENT_FILE>
```

## 备注
本项目借鉴了下列项目和网站的设计:
- [Artifical Analysis](https://artificialanalysis.ai/)
- [DeepSeek API Arena](https://deepseek.ai-infra.fun/)

本项目复用了 [OpenCompass](https://github.com/open-compass/opencompass)的代码来进行智能水平测试。
