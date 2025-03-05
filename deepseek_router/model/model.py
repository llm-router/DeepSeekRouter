import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pytz
from sqlalchemy import String, DateTime, create_engine, Engine, Text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase


class Metric(Enum):
    TTFT = 'ttft'
    PREFILL_RATE = 'prefill'
    DECODE_RATE = 'decode'
    OTPS = 'otps'
    OTPR = 'otpr'
    SUCCESS = 'success'
    GPQA = 'gpqa'
    AIME2024 = 'aime2024'


class Stat(Enum):
    Avg = 'avg'
    P50 = 'median'
    P90 = 'p90'
    P99 = 'p99'
    Raw = 'raw'

perf_metrics = [Metric.TTFT, Metric.PREFILL_RATE, Metric.DECODE_RATE, Metric.OTPS, Metric.OTPR, Metric.SUCCESS]
benchmark_metrics = [Metric.AIME2024, Metric.GPQA]

def convert_code_to_metric(metric: str) -> Optional[Metric]:
    try:
        return Metric(metric)
    except:
        return None


def convert_code_to_stat(stat: str) -> Optional[Stat]:
    try:
        return Stat(stat)
    except:
        return None


class Base(DeclarativeBase):
    pass


@dataclass
class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)


@dataclass
class ServiceTestResult(Base):
    __tablename__ = "perf_test_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64))
    provider_name: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    # prompt: Mapped[str] = mapped_column(Text(length=(2 ** 32) - 1))
    prompt_type: Mapped[str] = mapped_column(String(32), index=True)
    # reasoning_content: Mapped[str] = mapped_column(Text(length=(2 ** 32) - 1))
    # final_content: Mapped[str] = mapped_column(Text(length=(2 ** 32) - 1))
    success: Mapped[bool] = mapped_column()
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    time_to_first_token: Mapped[float] = mapped_column(nullable=True)
    time_to_first_result_token: Mapped[float] = mapped_column(nullable=True)
    total_time: Mapped[float] = mapped_column()
    input_tokens_count: Mapped[int] = mapped_column()
    output_tokens_count: Mapped[int] = mapped_column()
    reasoning_tokens_count: Mapped[int] = mapped_column()
    cached_tokens_count_billing: Mapped[int] = mapped_column()
    input_tokens_count_billing: Mapped[int] = mapped_column()
    output_tokens_count_billing: Mapped[int] = mapped_column()
    error: Mapped[str] = mapped_column(Text())

    def __init__(self,
                 provider_name: str,
                 model_name: str,
                 prompt: str,
                 prompt_type: str,
                 success: bool,
                 start_time: datetime,
                 end_time: datetime,
                 total_time: float,
                 input_tokens_count: int,
                 output_tokens_count: int,
                 reasoning_tokens_count: int,
                 cached_tokens_count_billing: int,
                 input_tokens_count_billing: int,
                 output_tokens_count_billing: int,
                 error: str,
                 time_to_first_token: Optional[datetime],
                 time_to_first_result_token: Optional[datetime],
                 reasoning_content: Optional[str],
                 final_content: Optional[str]):
        self.provider_name = provider_name
        self.model_name = model_name
        self.prompt = prompt
        self.prompt_type = prompt_type
        self.success = success
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.input_tokens_count = input_tokens_count
        self.output_tokens_count = output_tokens_count
        self.reasoning_tokens_count = reasoning_tokens_count
        self.cached_tokens_count_billing = cached_tokens_count_billing
        self.input_tokens_count_billing = input_tokens_count_billing
        self.output_tokens_count_billing = output_tokens_count_billing
        self.error = error
        self.time_to_first_token = time_to_first_token
        self.time_to_first_result_token = time_to_first_result_token
        self.reasoning_content = reasoning_content
        self.final_content = final_content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "prompt_type": self.prompt_type,
            "reasoning_content": self.reasoning_content,
            "final_content": self.final_content,
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "time_to_first_token": self.time_to_first_token,
            "time_to_first_result_token": self.time_to_first_result_token,
            "total_time": self.total_time,
            "input_tokens_count": self.input_tokens_count,
            "output_tokens_count": self.output_tokens_count,
            "reasoning_tokens_count": self.reasoning_tokens_count,
            "cached_tokens_count_billing": self.cached_tokens_count_billing,
            "input_tokens_count_billing": self.input_tokens_count_billing,
            "output_tokens_count_billing": self.output_tokens_count_billing,
            "error": self.error
        }

    def to_list(self) -> List[any]:
        return [
            self.job_id,
            self.provider_name,
            self.model_name,
            self.prompt_type,
            self.success,
            pytz.UTC.localize(self.start_time).isoformat(),
            pytz.UTC.localize(self.end_time).isoformat(),
            self.time_to_first_token,
            self.time_to_first_result_token,
            self.total_time,
            self.input_tokens_count,
            self.output_tokens_count,
            self.reasoning_tokens_count,
            self.cached_tokens_count_billing,
            self.input_tokens_count_billing,
            self.output_tokens_count_billing,
            self.error
        ]


@dataclass
class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    benchmark: Mapped[str] = mapped_column(String(64), index=True)
    success: Mapped[bool] = mapped_column()
    metric: Mapped[str] = mapped_column(String(32))
    stat: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column()
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    results_json: Mapped[str] = mapped_column(Text(length=(2 ** 32) - 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'provider_name': self.provider_name,
            'model_name': self.model_name,
            'benchmark': self.benchmark,
            'success': self.success,
            'metric': self.metric,
            'stat': self.stat,
            'value': self.value,
            'start_time': self.start_time,
            'results_json': self.results_json
        }


def initialize_engine() -> Engine:
    db_file_path = Path(__file__).parent / "../../config/db.json"
    with db_file_path.open("r") as db_file:
        db_config = json.load(db_file)
    print("Connecting to ", db_config['conn_str'])
    return create_engine(db_config['conn_str'])


def initialize_tables(engine):
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    my_engine = initialize_engine()
    initialize_tables(my_engine)
