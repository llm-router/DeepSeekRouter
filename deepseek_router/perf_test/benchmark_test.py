import argparse
import json
import os
from pathlib import Path
import re
from datetime import datetime
import subprocess
import warnings
from deepseek_router.model.model import BenchmarkResult

root_path = Path(__file__).parent.parent.parent

cron_file_path = Path(__file__).parent / "../../config/cron.json"
with cron_file_path.open("r") as cron_file:
    cron_config = json.load(cron_file)

services_file_path = Path(__file__).parent / "../../data/services.json"
with services_file_path.open("r") as services_file:
    services = json.load(services_file)

models_file_path = Path(__file__).parent / "../../data/models.json"
with models_file_path.open("r") as models_file:
    models = json.load(models_file)

keys_file_path = Path(__file__).parent / "../../config/keys.json"
with keys_file_path.open("r") as keys_file:
    keys = json.load(keys_file)

dataset_file_path = Path(__file__).parent / "../../data/datasets.json"
with dataset_file_path.open("r") as dataset_file:
    datasets = json.load(dataset_file)

def get_latest_subdir(directory):
    pattern = r'^\d{8}_\d{6}$'
    valid_subdirs = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and re.match(pattern, subdir):
            valid_subdirs.append(subdir)

    if not valid_subdirs:
        return None

    def convert_to_datetime(subdir_name):
        date_str, time_str = subdir_name.split('_')
        combined_str = f'{date_str} {time_str}'
        return datetime.strptime(combined_str, '%Y%m%d %H%M%S')

    sorted_subdirs = sorted(valid_subdirs, key=convert_to_datetime, reverse=True)
    return sorted_subdirs[0]

class BenchmarkTest:
    def __init__(self, model, provider, dataset):
        self.model = model
        self.provider = provider
        self.dataset = dataset
    
    def run(self, debug=False):
        if self.model not in models:
            print(f"Model {self.model} not found.")
            exit(1)
        
        service_dict = services[self.model]
        if self.provider not in service_dict:
            print(f"Provider {self.provider} not found.")
            exit(1)

        service = service_dict[self.provider]
        api_base = service['base_url']

        dataset_path = datasets[self.dataset]['path']
        print(f"Dataset path: {dataset_path}")
        dataset_name = datasets[self.dataset]['name']
        print(f"Dataset name: {dataset_name}")

        if keys[self.provider]['default'] == '':
            print(f"API key for provider {self.provider} not found.")
            exit(1)

        # Generate the model config for opencompass to use
        content = f'''
from opencompass.models.openai_api_simple import SimpleOpenAI
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='{self.model}',
        type=SimpleOpenAI,
        path='{service['model']}',
        api_key='{keys[self.provider]['default']}',  
        base_url='{api_base}',
        provider='{self.provider}',
        max_concurrency={service['max_concurrency']},
        timeout={models[self.model]['timeout']},
        tokenizer='{models[self.model]['tokenizer']}',
        generation_kwargs = {{
            'temperature': {models[self.model]['temperature']},
            'top_p': {models[self.model]['top_p']}, 
            'verbose': {debug}
        }},
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len={service['max_tokens']},
        max_seq_len={service['max_tokens']},
        batch_size=8),
]
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.{dataset_path} import {dataset_name}
'''
        
        # Pick the first dataset for debugging
        if debug:
            content += f'''
datasets = {dataset_name}[:1]
datasets[0]['abbr'] = 'demo_' + datasets[0]['abbr']
datasets[0]['reader_cfg']['test_range'] = '[0:3]'
'''
        else:
            content += f'''
datasets = {dataset_name}
'''
        # Write the model config to a tmp file
        log_path = cron_config['log_dir']
        config_path = os.path.join(log_path, 'benchmark_model_configs', f'{self.model}-{self.provider}-{self.dataset}.py')
        os.makedirs(os.path.join(log_path, 'benchmark_model_configs'), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(content)

        # Run the evaluation using opencompass
        start_time = datetime.now()
        output_path = os.path.join(log_path, 'benchmark_logs', self.provider, self.dataset)
        command = ["opencompass", config_path, "-w", output_path]
        if debug:
            command.append('--debug')
        print("Running evaluation...")
        subprocess.run(command)
        print("Evaluation completed.")

        # Get the latest subdir in the output directory
        latest_subdir = get_latest_subdir(output_path)
        if latest_subdir is None:
            print("No evaluation results found.")
            exit(1)
        result_path = os.path.join(output_path, latest_subdir, 'results', self.model)
        prediction_path = os.path.join(output_path, latest_subdir, 'predictions', self.model)

        # Get the accuracy from the result file
        datasets_results = {}
        metric = "accuracy"
        acc_sum = 0.0
        total_sum = 0.0
        for result_file in os.listdir(result_path):
            with open(os.path.join(result_path, result_file), "r") as f:
                result = json.load(f)
                print("Get result from file: ", result_file)
                if 'accuracy' in result:
                    metric = "accuracy"
                elif 'pass@1' in result:
                    metric = "pass@1"
                else:
                    metric = list(result.keys())[0]
                    warnings.warn(f"Unknown metric {metric} found in the result file.")
                with open(os.path.join(prediction_path,result_file), "r") as f:
                    predictions = json.load(f)
                num = len(predictions)
                acc_sum += result[metric] * num
                total_sum += num
                datasets_results[result_file] = result[metric]
                if len(result) > 1:
                    warnings.warn("More than one result item found in the result file.")
        
        if total_sum == 0.0:
            warnings.warn("No result found.")
            return None

        return BenchmarkResult(
            provider_name=self.provider,
            model_name=self.model,
            benchmark= ('demo_' if debug else '') + self.dataset,
            success=True,
            metric=metric,
            stat='average',
            value=acc_sum / total_sum,
            start_time=start_time,
            results_json=json.dumps(datasets_results)
        )

# Note: The following code is for testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the performance of specified model on the specified dataset.')
    parser.add_argument('--model', type=str, default='deepseek-r1' , help='The model to evaluate.')
    parser.add_argument('--provider', type=str, default='aliyun', help='The provider of the model.')
    parser.add_argument('--dataset', type=str, default='aime2024', help='The dataset to evaluate the model on.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode.')
    
    args = parser.parse_args()

    print(f"Evaluating model {args.model} on dataset {args.dataset}...")
    print(f"Provider: {args.provider}")

    test = BenchmarkTest(args.model, args.provider, args.dataset)
    result = test.run(args.debug)
    print(result)
    