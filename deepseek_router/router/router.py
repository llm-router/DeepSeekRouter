import json
import os
import statistics
import sys
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from deepseek_router.utils.utils import convert_currency

"""
Example input:
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
    }
  ],
  "order": [
    {
      "metric": "output_price"
    }
  ]
}

Supported metrics: ttft, prefill, decode, otps, otpr, success, gpqa, aime2024, output_price, input_price
Supported stats: avg, median, p90, p99
"""


def generate_intermediate_result(data_path: str, model: str) -> DataFrame:
    columns = ["JOB_ID", "PROVIDER", "MODEL", "PROMPT_TYPE", "SUCCESS", "START_TIME", "END_TIME", "TIME_TO_FIRST_TOKEN",
               "TIME_TO_FIRST_RESULT_TOKEN", "TOTAL_TIME", "INPUT_TOKENS_COUNT", "OUTPUT_TOKENS_COUNT",
               "REASONING_TOKENS_COUNT", "CACHED_TOKENS_COUNT_BILLING", "INPUT_TOKENS_COUNT_BILLING",
               "OUTPUT_TOKENS_COUNT_BILLING", "ERROR"]
    df = pd.read_csv(os.path.join(data_path, 'perf_results.csv'), header=None, names=columns)
    model_df = df[df['MODEL'] == model]
    providers = model_df['PROVIDER'].unique()
    rows = []
    columns = [
        'provider',
        'ttft_avg', 'ttft_median', 'ttft_p90', 'ttft_p99',
        'prefill_avg', 'prefill_median', 'prefill_p90', 'prefill_p99',
        'decode_avg', 'decode_median', 'decode_p90', 'decode_p99',
        'otps_avg', 'otps_median', 'otps_p90', 'otps_p99',
        'otpr_avg', 'otpr_median', 'otpr_p90', 'otpr_p99',
        'success', 'gpqa', 'aime2024', 'livecodebench', 'mmlu', 'ceval', 'output_price', 'input_price'
    ]
    for provider in providers:
        rows.append([provider] + ([0] * (len(columns) - 1)))
    ret_df = pd.DataFrame(np.array(rows), columns=columns)
    for provider in providers:
        provider_df = model_df[model_df['PROVIDER'] == provider]
        ttft = provider_df['TIME_TO_FIRST_TOKEN']
        ret_df.loc[ret_df.provider == provider, 'ttft_avg'] = statistics.fmean(ttft)
        ret_df.loc[ret_df.provider == provider, 'ttft_median'] = np.percentile(ttft, 50)
        ret_df.loc[ret_df.provider == provider, 'ttft_p90'] = np.percentile(ttft, 90)
        ret_df.loc[ret_df.provider == provider, 'ttft_p99'] = np.percentile(ttft, 99)
        prefill = provider_df['INPUT_TOKENS_COUNT'] / provider_df['TIME_TO_FIRST_TOKEN']
        ret_df.loc[ret_df.provider == provider, 'prefill_avg'] = statistics.fmean(prefill)
        ret_df.loc[ret_df.provider == provider, 'prefill_median'] = np.percentile(prefill, 50)
        ret_df.loc[ret_df.provider == provider, 'prefill_p90'] = np.percentile(prefill, 10)
        ret_df.loc[ret_df.provider == provider, 'prefill_p99'] = np.percentile(prefill, 1)
        decode = provider_df['OUTPUT_TOKENS_COUNT'] / (provider_df['TOTAL_TIME'] - provider_df['TIME_TO_FIRST_TOKEN'])
        ret_df.loc[ret_df.provider == provider, 'decode_avg'] = statistics.fmean(decode)
        ret_df.loc[ret_df.provider == provider, 'decode_median'] = np.percentile(decode, 50)
        ret_df.loc[ret_df.provider == provider, 'decode_p90'] = np.percentile(decode, 10)
        ret_df.loc[ret_df.provider == provider, 'decode_p99'] = np.percentile(decode, 1)
        otps = provider_df['OUTPUT_TOKENS_COUNT'] / provider_df['TOTAL_TIME']
        ret_df.loc[ret_df.provider == provider, 'otps_avg'] = statistics.fmean(otps)
        ret_df.loc[ret_df.provider == provider, 'otps_median'] = np.percentile(otps, 50)
        ret_df.loc[ret_df.provider == provider, 'otps_p90'] = np.percentile(otps, 10)
        ret_df.loc[ret_df.provider == provider, 'otps_p99'] = np.percentile(otps, 1)
        otpr = provider_df['OUTPUT_TOKENS_COUNT']
        ret_df.loc[ret_df.provider == provider, 'otpr_avg'] = statistics.fmean(otpr)
        ret_df.loc[ret_df.provider == provider, 'otpr_median'] = np.percentile(otpr, 50)
        ret_df.loc[ret_df.provider == provider, 'otpr_p90'] = np.percentile(otpr, 10)
        ret_df.loc[ret_df.provider == provider, 'otpr_p99'] = np.percentile(otpr, 1)
        success = provider_df['SUCCESS']
        ret_df.loc[ret_df.provider == provider, 'success'] = statistics.fmean(success)
        input_price = services[model][provider]['input_price_per_million_token'] \
            if 'input_price_per_million_token' in services[model][provider] else 0
        output_price = services[model][provider]['output_price_per_million_token'] \
            if 'output_price_per_million_token' in services[model][provider] else 0
        input_price = convert_currency(input_price, services[model][provider]['currency'], 'CNY')
        output_price = convert_currency(output_price, services[model][provider]['currency'], 'CNY')
        ret_df.loc[ret_df.provider == provider, 'input_price'] = input_price
        ret_df.loc[ret_df.provider == provider, 'output_price'] = output_price

    benchmark_df = pd.read_csv(os.path.join(args.data_dir, 'benchmark_results.csv'))
    for _, row in benchmark_df.iterrows():
        if (row['model_name'] == args.model and row['success'] == True and row['benchmark'] in columns):
            ret_df.loc[ret_df.provider == row["provider_name"], row["benchmark"]] = row["value"]

    return ret_df


def apply_filters(df: DataFrame, filters: List[Dict], order: List[Dict]) -> DataFrame:
    ret_df = df
    for filter_condition in filters:
        column = filter_condition['metric']
        if 'stat' in filter_condition:
            column = column + '_' + filter_condition['stat']
        if 'min' in filter_condition:
            ret_df = ret_df[ret_df[column] >= filter_condition['min']]
        if 'max' in filter_condition:
            ret_df = ret_df[ret_df[column] <= filter_condition['max']]
    asc_columns = ['ttft_avg', 'ttft_median', 'ttft_p90', 'ttft_p99', 'output_price', 'input_price']
    columns = []
    columns_asc = []
    for order_condition in order:
        column = order_condition['metric']
        if 'stat' in order_condition:
            column = column + '_' + order_condition['stat']
        columns.append(column)
        columns_asc.append(column in asc_columns)
    return ret_df.sort_values(columns, ascending=columns_asc)


if __name__ == "__main__":
    cron_file_path = Path(__file__).parent / "../../config/cron.json"
    with cron_file_path.open("r") as cron_file:
        cron_config = json.load(cron_file)
    services_file_path = Path(__file__).parent / "../../data/services.json"
    with services_file_path.open("r") as services_file:
        services = json.load(services_file)

    parser = argparse.ArgumentParser(
        description='Filter the right providers for you.')
    parser.add_argument('--requirement', type=str, help='Path to your requirement file', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to your evaluation date',
                        default=cron_config['log_dir'])
    parser.add_argument('--model', type=str, help='Which model you are targeting',
                        default='deepseek-r1')
    args = parser.parse_args()
    requirement_json = None
    if os.path.exists(args.requirement):
        with open(args.requirement, "r") as requirement_file:
            requirement_json = json.load(requirement_file)
    if requirement_json is None:
        print("Requirement file does not exist or is not a JSON!")
        sys.exit(-1)

    user_filters = requirement_json['filters']
    user_order = requirement_json['order']
    data_frame1 = generate_intermediate_result(args.data_dir, args.model)
    data_frame2 = apply_filters(data_frame1, user_filters, user_order)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data_frame2)
