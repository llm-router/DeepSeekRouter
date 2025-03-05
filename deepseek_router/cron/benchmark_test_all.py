import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session

from deepseek_router.perf_test.benchmark_test import BenchmarkTest
from deepseek_router.model.model import initialize_engine, BenchmarkResult

def benchmark_evaluate(model, provider, dataset, debug=False):
    test = BenchmarkTest(model, provider, dataset)
    result = test.run(debug)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the performance of specified model on the specified dataset.')
    parser.add_argument('--model', type=str, default='deepseek-r1' , help='The model to evaluate.')
    parser.add_argument('--dataset', type=str, default='aime2024', help='The dataset to evaluate the model on.')
    parser.add_argument('--provider', type=str, default='', help='The provider of the model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('-o', '--output', type=str, default='csv', choices=['db', 'csv', 'both'], help='The storage format.')
    args = parser.parse_args()

    cron_file_path = Path(__file__).parent / "../../config/cron.json"
    with cron_file_path.open("r") as cron_file:
        cron_config = json.load(cron_file)

    models_file_path = Path(__file__).parent / "../../data/models.json"
    with models_file_path.open("r") as models_file:
        models = json.load(models_file)

    services_file_path = Path(__file__).parent / "../../data/services.json"
    with services_file_path.open("r") as services_file:
        services = json.load(services_file)

    keys_file_path = Path(__file__).parent / "../../config/keys.json"
    with keys_file_path.open("r") as keys_file:
        keys = json.load(keys_file)

    model_services = services[args.model]
    model_services = [service for service in model_services if keys[service]['default'] != '']

    # If provider is specified, only evaluate the model on the specified provider
    if args.provider != '':
        provider_list = args.provider.split(',')
        model_services = [service for service in model_services if service in provider_list]
        
    start_time = datetime.now()
    write_to_db = args.output == 'db' or args.output == 'both'
    write_to_csv = args.output == 'csv' or args.output == 'both'

    if write_to_db:
        my_engine = initialize_engine()

    if write_to_csv:
        raw_table_path = os.path.join(cron_config['log_dir'], f'benchmark_results.csv')
        if not os.path.exists(raw_table_path):
            os.makedirs(cron_config['log_dir'], exist_ok=True)
            with open(raw_table_path, "a") as raw_table:
                raw_table.write("model_name,provider_name,benchmark,success,start_time,metric,stat,value,results_json\n")

    # Distribute the benchmark evaluation to multiple threads
    with ThreadPoolExecutor(max_workers=len(model_services)) as executor:
        future_to_service = {executor.submit(benchmark_evaluate, args.model, provider, args.dataset, args.debug): provider for provider in model_services}
        for future in as_completed(future_to_service):
            provider = future_to_service[future]
            try:
                result = future.result()
                print(f"Success: Provider: {provider}, Result: {result}")
            except Exception as exc:
                result = BenchmarkResult(
                    model_name=args.model,
                    provider_name=provider,
                    benchmark=args.dataset,
                    success=False,
                    start_time=start_time,
                    metric="None",
                    stat="None",
                    value=-1.0,
                    results_json="None"
                )
                print(f"Failed: Provider: {provider}, Exception: {exc}")
                
            if write_to_db:
                with Session(my_engine) as session:
                    session.add(result)
                    session.commit()
                    print("Failed result saved to database.")
            if write_to_csv:
                with open(raw_table_path, "a") as raw_table:
                    raw_table.write(f"{result.benchmark},{result.model_name},{result.provider_name},{result.benchmark},{result.success},{result.start_time},{result.metric},{result.stat},{result.value},{result.results_json}\n")
        