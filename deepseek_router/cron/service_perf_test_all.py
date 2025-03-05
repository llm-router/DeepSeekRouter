#!/usr/bin/python
import csv
import json
import os.path
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from deepseek_router.perf_test.service_test import ServiceTest
from deepseek_router.perf_test.prompt_generator import ShareGPTPromptGenerator, LEvalPromptGenerator
from deepseek_router.model.model import initialize_engine, Job


def generate_prompts(model_name: str) -> List[Tuple[str, str, Optional[List]]]:
    share_gpt_generator = ShareGPTPromptGenerator(load_local=True)
    leval_generator = LEvalPromptGenerator(load_local=True)
    ret = list()
    single_round_prompt = share_gpt_generator.generate(model_name, 0, sys.maxsize, 128)
    ret.append((single_round_prompt, 'sharegpt:single_round', None))
    multi_round_prompt, multi_round_context = (
        share_gpt_generator.generate_multi_round(model_name, 0, sys.maxsize, 128))
    ret.append((multi_round_prompt, 'sharegpt:multi_round', multi_round_context))
    long_prompt = (
        leval_generator.generate(model_name, 0, sys.maxsize, output_len=models[model]['max_tokens']))
    ret.append((long_prompt, 'leval', None))
    return ret


def run_perf_test_for_provider(model_name: str, provider_name: str, prompts: List[Tuple[str, str, Optional[List]]],
                               job_id: str, engine: Engine):
    if export_to_db:
        session = Session(engine)
    service_test = None
    while service_test is None:
        try:
            service_test = ServiceTest(model_name, provider_name, keys[provider_name]['default'])
        except:
            logging.error("An error occurred", exc_info=True)
    for prompt in prompts:
        try:
            result = service_test.run_test(prompt=prompt[0], prompt_type=prompt[1], context=prompt[2])
            result.job_id = job_id
            with open(os.path.join(cron_config['log_dir'], my_job_id,
                                   f'{model_name}-{provider_name}-{prompt[1]}.json'), "w") as file:
                log = {
                    "prompt": result.prompt,
                    "reasoning_content": result.reasoning_content,
                    "final_content": result.final_content
                }
                json.dump(log, file, indent=4)
            if export_to_csv:
                with open(os.path.join(cron_config['log_dir'], 'perf_results.csv'), "a", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(result.to_list())
            if export_to_db:
                session.add(result)
                session.commit()
        except:
            logging.error("An error occurred", exc_info=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate the performance of all providers.')
    parser.add_argument('--output', type=str, help='Output to [db], [csv], or [both].', default='csv')
    args = parser.parse_args()
    output_destination = args.output
    if output_destination == 'db':
        export_to_db = True
        export_to_csv = False
    elif output_destination == 'csv':
        export_to_db = False
        export_to_csv = True
    elif output_destination == 'both':
        export_to_db = True
        export_to_csv = True
    else:
        print("Invalid output format!")
        sys.exit(-1)

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

    my_engine = initialize_engine()
    my_job_id = str(uuid.uuid4())
    Path(os.path.join(cron_config['log_dir'], my_job_id)).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cron_config['log_dir'], my_job_id, 'error.log'),
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    with Session(my_engine) as my_session:
        job = Job(id=my_job_id, type='perf', start_time=datetime.now())
        my_session.add(job)
        my_session.commit()
    for model in cron_config['perf_models']:
        my_prompts = generate_prompts(model)
        futures = list()
        with ThreadPoolExecutor(max_workers=cron_config['max_workers_per_job']) as executor:
            for provider, provider_config in services[model].items():
                if provider in keys and len(keys[provider]['default']) != 0:
                    futures.append(executor.submit(run_perf_test_for_provider,
                                                   model_name=model,
                                                   provider_name=provider,
                                                   prompts=my_prompts,
                                                   job_id=my_job_id,
                                                   engine=my_engine))
            for future in as_completed(futures):
                future.result()
    with Session(my_engine) as my_session:
        my_session.query(Job).filter(Job.id == my_job_id).update({'end_time': datetime.now()})
        my_session.commit()
