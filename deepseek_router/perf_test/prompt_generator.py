from datetime import datetime
import os
import random
from typing import List, Optional, Tuple

import datasets
import json
from abc import abstractmethod, ABC
from pathlib import Path

from datasets import Dataset
from openai import OpenAI
from transformers import AutoTokenizer

models_file_path = Path(__file__).parent / "../../data/models.json"
with models_file_path.open("r") as models_file:
    models = json.load(models_file)

generator_file_path = Path(__file__).parent / "../../config/prompt_generator.json"
with generator_file_path.open("r") as generator_file:
    generator_config = json.load(generator_file)


class PerformancePromptGenerator(ABC):
    @abstractmethod
    def generate(self, target_model: str, input_len_min: int, input_len_max: int, output_len: int) -> Optional[str]:
        raise NotImplementedError


class SimpleLLMPerformancePromptGenerator(PerformancePromptGenerator, ABC):

    def __init__(self):
        model_metadata = generator_config['model']
        self.client = OpenAI(
            base_url=model_metadata['base_url'],
            api_key=model_metadata['api_key']
        )
        self.reasoning_inline = model_metadata['reasoning_inline']
        self.reasoning_start = model_metadata['reasoning_start'] if self.reasoning_inline else None
        self.reasoning_end = model_metadata['reasoning_end'] if self.reasoning_inline else None
        self.model = model_metadata['model']
        self.init_query = ('Generate a random question of $min$ to $max$ tokens without answering it. '
                           'Use the format <q>question</q>.')
        self.improve_query = ('Ask a follow-up question of $min$ to $max$ tokens to the given questions: $questions$.'
                              'Use the format <q>question</q>.')
        self.prefix_query = 'Answer the following questions in $output$ tokens: '

    def _generate_question(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user",
                 "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip().replace('\n', '')
        if self.reasoning_inline:
            content = content.split(self.reasoning_end)[-1]
        if content.startswith('<q>') and content.endswith('</q>'):
            return content
        else:
            return self._generate_question(prompt)

    def generate(self, target_model: str, input_len_min: int, input_len_max: int, output_len: int) -> Optional[str]:
        tokenizer = AutoTokenizer.from_pretrained(models[target_model]['tokenizer'])
        questions = ''
        questions_tokens_count = 0
        while questions_tokens_count < input_len_min:
            new_min = input_len_min - questions_tokens_count
            new_max = input_len_max - questions_tokens_count
            query = self.init_query if questions_tokens_count == 0 else self.improve_query
            new_question = self._generate_question(query.replace('$min$', str(new_min))
                                                   .replace('$max$', str(new_max))
                                                   .replace('$questions$', questions))
            new_question_tokens = tokenizer(new_question)
            new_question_tokens_count = len(new_question_tokens['input_ids'])
            if questions_tokens_count + new_question_tokens_count > input_len_max:
                continue
            questions += '' if questions_tokens_count == 0 else '\n'
            questions += new_question
            questions_tokens_count += new_question_tokens_count
            print(questions)
            print(questions_tokens_count)
        prefix = self.prefix_query.replace('$output$', str(output_len))
        return prefix + questions


class DatasetPromptGenerator(PerformancePromptGenerator):

    @abstractmethod
    def generate_multiple(self, target_model: str, input_len_min: int, input_len_max: int,
                          output_len: int, num_prompts: int = -1) -> List[str]:
        raise NotImplementedError


class ShareGPTPromptGenerator(DatasetPromptGenerator):

    def __init__(self, load_local: bool = False):
        self.name = 'ShareGPT'
        self.path = 'shibing624/sharegpt_gpt4'
        self.context = 'Current time is $time$\n'
        self.prefix_query = 'Answer the following questions in $output$ tokens: '
        if load_local and os.path.exists(os.path.join(Path(__file__).parent / generator_config['datasets']['directory'],
                                                      self.name)):
            Path(Path(__file__).parent / generator_config['datasets']['directory']).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset = datasets.load_from_disk(save_path)
        else:
            self.load_remote()

    def load_remote(self):
        dataset = datasets.load_dataset(self.path)
        single_round_prompt_list = list()
        multi_round_prompt_list = list()
        multi_round_context_list = list()
        dataset_train = dataset['train']
        for row in dataset_train:
            messages = row['conversations']
            single_round_prompt_list.append(messages[0]['value'])
            idx = -1
            while messages[idx]['from'] != 'human':
                idx -= 1
            multi_round_prompt_list.append(messages[idx]['value'])
            multi_round_context = list()
            for message in messages[:idx]:
                new_message = {
                    'role': 'user' if message['from'] == 'human' else 'assistant',
                    'content': message['value']
                }
                multi_round_context.append(new_message)
            multi_round_context_list.append(multi_round_context)
        dataset_train = dataset_train.add_column('single_round_prompt', single_round_prompt_list)
        dataset_train = dataset_train.add_column('multi_round_prompt', multi_round_prompt_list)
        dataset_train = dataset_train.add_column('multi_round_context', multi_round_context_list)
        dataset_train = dataset_train.remove_columns('conversations')
        save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
        dataset_train.save_to_disk(save_path)
        self.dataset = dataset_train

    def _count_token(self, target_model: str, re_count: bool = False):
        need_save = False
        tokenizer = AutoTokenizer.from_pretrained(models[target_model]['tokenizer'])
        if re_count:
            if f'single_round_tokens_{target_model}' not in self.dataset.column_names:
                self.dataset = self.dataset.remove_columns(f'single_round_tokens_{target_model}')
            if f'multi_round_tokens_{target_model}' not in self.dataset.column_names:
                self.dataset = self.dataset.remove_columns(f'multi_round_tokens_{target_model}')
        if f'single_round_tokens_{target_model}' not in self.dataset.column_names:
            need_save = True
            single_round_tokens_list = list()
            for row in self.dataset:
                messages = [
                    {'role': 'user', 'content': row['single_round_prompt']}
                ]
                single_round_tokens_list.append(len(tokenizer.apply_chat_template(messages, tokenize=True)))
            self.dataset = self.dataset.add_column(f'single_round_tokens_{target_model}', single_round_tokens_list)
        if f'multi_round_tokens_{target_model}' not in self.dataset.column_names:
            need_save = True
            multi_round_tokens_list = list()
            for row in self.dataset:
                messages = list(row['multi_round_context'])
                role = 'user'
                valid = True
                for message in messages:
                    if message['role'] != role:
                        valid = False
                        break
                    role = 'user' if role == 'assistant' else 'assistant'
                if valid:
                    messages.append({'role': 'user', 'content': row['multi_round_prompt']})
                    multi_round_tokens_list.append(len(tokenizer.apply_chat_template(messages, tokenize=True)))
                else:
                    multi_round_tokens_list.append(-1)
            self.dataset = self.dataset.add_column(f'multi_round_tokens_{target_model}', multi_round_tokens_list)
        if need_save:
            self.dataset = Dataset.from_dict(self.dataset.to_dict())
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset.save_to_disk(save_path)

    def generate(self, target_model: str, input_len_min: int, input_len_max: int, output_len: int) -> Optional[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'single_round_tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        prefix = (self.context.replace('$time$', date_time) +
                  self.prefix_query.replace('$output$', str(output_len)))
        if len(filtered_dataset) > 0:
            return prefix + random.choice(filtered_dataset['single_round_prompt'])
        else:
            return None

    def generate_multiple(self, target_model: str, input_len_min: int, input_len_max: int,
                          output_len: int, num_prompts: int = -1) -> List[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'single_round_tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        prefix = (self.context.replace('$time$', date_time) +
                  self.prefix_query.replace('$output$', str(output_len)))
        if num_prompts == -1:
            return [prefix + prompt for prompt in filtered_dataset['single_round_prompt']]
        else:
            return random.sample([prefix + prompt for prompt in filtered_dataset['single_round_prompt']],
                                 min(num_prompts, len(filtered_dataset)))

    def generate_multi_round(self, target_model: str, input_len_min: int, input_len_max: int,
                             output_len: int) -> Tuple[Optional[str], Optional[List]]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'multi_round_tokens_{target_model}'] <= input_len_max
                            and len(example['multi_round_context']) > 0)
        prefix = self.prefix_query.replace('$output$', str(output_len))
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        extra_context = self.context.replace('$time$', date_time)
        if len(filtered_dataset) > 0:
            sample = random.choice(filtered_dataset)
            context = list(sample['multi_round_context'])
            context[0]['content'] = extra_context + context[0]['content']
            return prefix + sample['multi_round_prompt'], context
        else:
            return None, None

    def generate_multi_round_multiple(self, target_model: str, input_len_min: int, input_len_max: int,
                                      output_len: int, num_prompts: int = -1) -> List[Tuple[str, List]]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'multi_round_tokens_{target_model}'] <= input_len_max
                            and len(example['multi_round_context']) > 0)
        prefix = self.prefix_query.replace('$output$', str(output_len))
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        extra_context = self.context.replace('$time$', date_time)
        if num_prompts == -1:
            ret = [(prefix + sample['multi_round_prompt'], list(sample['multi_round_context']))
                   for sample in filtered_dataset]
            for element in ret:
                element[1][0]['content'] = extra_context + element[1][0]['content']
        else:
            samples = random.sample(filtered_dataset, min(num_prompts, len(filtered_dataset)))
            ret = [(prefix + sample['multi_round_prompt'], list(sample['multi_round_context']))
                   for sample in samples]
            for element in ret:
                element[1][0]['content'] = extra_context + element[1][0]['content']
        return ret


class LongBenchPromptGenerator(DatasetPromptGenerator):

    def __init__(self, load_local: bool = False):
        self.name = 'LongBench-v2'
        self.path = 'THUDM/LongBench-v2'
        self.template_query = ('Current time is $time$\n'
                               'Answer the following question and elaborate in $output$ tokens '
                               'based on the given context.\n'
                               '[Question]: $question$ Your choices are: '
                               'A. $choice_A$; B. $choice_B$; C. $choice_C$; D. $choice_D$\n'
                               '[Context]: $context$\n')
        if load_local and os.path.exists(os.path.join(Path(__file__).parent / generator_config['datasets']['directory'],
                                                      self.name)):
            Path(Path(__file__).parent / generator_config['datasets']['directory']).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset = datasets.load_from_disk(save_path)
        else:
            self.load_remote()

    def load_remote(self):
        dataset = datasets.load_dataset(self.path)
        dataset_train = dataset['train']
        dataset_train = dataset_train.remove_columns('answer')
        save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
        dataset_train.save_to_disk(save_path)
        self.dataset = dataset_train

    def _count_token(self, target_model: str, re_count: bool = False):
        need_save = False
        tokenizer = AutoTokenizer.from_pretrained(models[target_model]['tokenizer'])
        if re_count:
            if f'tokens_{target_model}' not in self.dataset.column_names:
                self.dataset = self.dataset.remove_columns(f'tokens_{target_model}')
        if f'tokens_{target_model}' not in self.dataset.column_names:
            need_save = True
            tokens_list = list()
            for row in self.dataset:
                messages = [
                    {'role': 'user', 'content': row['context'] + row['question'] + row['choice_A'] + row['choice_B']
                                                + row['choice_C'] + row['choice_D']}
                ]
                tokens_list.append(len(tokenizer.apply_chat_template(messages, tokenize=True)))
            self.dataset = self.dataset.add_column(f'tokens_{target_model}', tokens_list)
        if need_save:
            self.dataset = Dataset.from_dict(self.dataset.to_dict())
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset.save_to_disk(save_path)

    def generate(self, target_model: str, input_len_min: int, input_len_max: int, output_len: int) -> Optional[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        query = self.template_query.replace('$output$', str(output_len)).replace('$time$', date_time)
        if len(filtered_dataset) > 0:
            choice = random.choice(filtered_dataset)
            query = (query.replace('$context$', choice['context'])
                     .replace('$question$', choice['question'])
                     .replace('$choice_A$', choice['choice_A'])
                     .replace('$choice_B$', choice['choice_B'])
                     .replace('$choice_C$', choice['choice_C'])
                     .replace('$choice_D$', choice['choice_D']))
            return query
        else:
            return None

    def generate_multiple(self, target_model: str, input_len_min: int, input_len_max: int,
                          output_len: int, num_prompts: int = -1) -> List[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        query = self.template_query.replace('$output$', str(output_len)).replace('$time$', date_time)
        if num_prompts == -1:
            return [query.replace('$context$', sample['context'])
                    .replace('$question$', sample['question'])
                    .replace('$choice_A$', sample['choice_A'])
                    .replace('$choice_B$', sample['choice_B'])
                    .replace('$choice_C$', sample['choice_C'])
                    .replace('$choice_D$', sample['choice_D'])
                    for sample in filtered_dataset]
        else:
            return random.sample([query.replace('$context$', sample['context'])
                                 .replace('$question$', sample['question'])
                                 .replace('$choice_A$', sample['choice_A'])
                                 .replace('$choice_B$', sample['choice_B'])
                                 .replace('$choice_C$', sample['choice_C'])
                                 .replace('$choice_D$', sample['choice_D']) for sample in filtered_dataset],
                                 min(num_prompts, len(filtered_dataset)))


class LEvalPromptGenerator(DatasetPromptGenerator):

    def __init__(self, load_local: bool = False):
        self.name = 'LEval'
        self.path = 'L4NLP/LEval'
        self.subsets = ["coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", "codeU", "sci_fi",
                        "financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "multidoc_qa",
                        "narrative_qa", "natural_question", "news_summ", "paper_assistant", "patent_summ",
                        "review_summ", "scientific_qa", "tv_show_summ"]
        self.template_query = ('Current time is $time$\n'
                               'Answer the following question and elaborate in $output$ tokens '
                               'based on the given context.\n'
                               '[Question]: $question$\n'
                               '[Context]: $context$\n')
        if load_local and os.path.exists(os.path.join(Path(__file__).parent / generator_config['datasets']['directory'],
                                                      self.name)):
            Path(Path(__file__).parent / generator_config['datasets']['directory']).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset = datasets.load_from_disk(save_path)
        else:
            self.load_remote()

    def load_remote(self):
        dataset = None
        for subset in self.subsets:
            if dataset is None:
                dataset = datasets.load_dataset(self.path, subset, split='test')
            else:
                dataset = datasets.concatenate_datasets([dataset,
                                                         datasets.load_dataset(self.path, subset, split='test')])
        dataset_dict = {'instruction': list(), 'input': list()}
        for row in dataset:
            for instruction in row['instructions']:
                dataset_dict['instruction'].append(instruction)
                dataset_dict['input'].append(row['input'])
        dataset = Dataset.from_dict(dataset_dict)
        save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
        dataset.save_to_disk(save_path)
        self.dataset = dataset

    def _count_token(self, target_model: str, re_count: bool = False):
        need_save = False
        tokenizer = AutoTokenizer.from_pretrained(models[target_model]['tokenizer'])
        if re_count:
            if f'tokens_{target_model}' not in self.dataset.column_names:
                self.dataset = self.dataset.remove_columns(f'tokens_{target_model}')
        if f'tokens_{target_model}' not in self.dataset.column_names:
            need_save = True
            tokens_list = list()
            for row in self.dataset:
                messages = [
                    {'role': 'user', 'content': row['instruction'] + row['input']}
                ]
                tokens_list.append(len(tokenizer.apply_chat_template(messages, tokenize=True)))
            self.dataset = self.dataset.add_column(f'tokens_{target_model}', tokens_list)
        if need_save:
            self.dataset = Dataset.from_dict(self.dataset.to_dict())
            save_path = os.path.join(Path(__file__).parent / generator_config['datasets']['directory'], self.name)
            self.dataset.save_to_disk(save_path)

    def generate(self, target_model: str, input_len_min: int, input_len_max: int, output_len: int) -> Optional[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        query = self.template_query.replace('$output$', str(output_len)).replace('$time$', date_time)
        if len(filtered_dataset) > 0:
            choice = random.choice(filtered_dataset)
            query = (query.replace('$context$', choice['input'])
                     .replace('$question$', choice['instruction']))
            return query
        else:
            return None

    def generate_multiple(self, target_model: str, input_len_min: int, input_len_max: int,
                          output_len: int, num_prompts: int = -1) -> List[str]:
        self._count_token(target_model)
        filtered_dataset = self.dataset.filter(
            lambda example: input_len_min <= example[f'tokens_{target_model}'] <= input_len_max)
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        query = self.template_query.replace('$output$', str(output_len)).replace('$time$', date_time)
        if num_prompts == -1:
            return [query.replace('$context$', sample['input'])
                    .replace('$question$', sample['instruction']) for sample in filtered_dataset]
        else:
            return random.sample([query.replace('$context$', sample['input'])
                                 .replace('$question$', sample['instruction']) for sample in filtered_dataset],
                                 min(num_prompts, len(filtered_dataset)))


if __name__ == "__main__":
    # generator = SimpleLLMPerformancePromptGenerator()
    generator = ShareGPTPromptGenerator(load_local=True)
    # generator = LongBenchPromptGenerator(load_local=True)
    # generator = LEvalPromptGenerator(load_local=True)
    # print(generator.generate('deepseek-r1', input_len_min=0, input_len_max=100000, output_len=100))
    print(generator.generate_multi_round('deepseek-r1', input_len_min=0, input_len_max=1000, output_len=100))
    # print(generator.generate_multiple('deepseek-r1', input_len_min=0, input_len_max=1000,
    #                                   output_len=100, num_prompts=2))
