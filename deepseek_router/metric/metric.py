from datetime import datetime, timedelta
import pytz
from typing import Union, Tuple, Optional, List

from sqlalchemy import Engine, Row
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import true

import statistics
import numpy as np

from deepseek_router.model.model import BenchmarkResult, Metric, Stat, ServiceTestResult


def generate_metric(metric: Metric, row: Row[Tuple]) -> Optional[Union[int, float]]:
    if metric == Metric.SUCCESS:
        return int(row[4])
    elif metric == Metric.OTPR:
        return row[1]
    elif metric == Metric.OTPS:
        return (row[1] / row[3]) if row[3] > 0 else None
    elif metric == Metric.TTFT:
        return row[2]
    elif metric == Metric.DECODE_RATE:
        return (row[1] / (row[3] - row[2])) if row[3] - row[2] > 0 else None
    elif metric == Metric.PREFILL_RATE:
        return (row[0] / row[2]) if row[2] > 0 else None
    else:
        return None


def generate_stat(metric: Metric, stat: Stat, data: List[Union[int, float]]) -> List[Union[int, float]]:
    if len(data) == 0 or stat == Stat.Raw:
        return data
    elif stat == Stat.Avg:
        return [statistics.fmean(data)]
    elif stat == Stat.P50:
        return [np.percentile(data, 50)]
    elif stat == Stat.P90:
        return [np.percentile(data, 90)] if metric == Metric.TTFT else [np.percentile(data, 10)]
    elif stat == Stat.P99:
        return [np.percentile(data, 99)] if metric == Metric.TTFT else [np.percentile(data, 1)]
    else:
        return data


class MetricQueryEngine:

    def __init__(self, engine: Engine):
        self.engine = engine

    def query_performance_metric(self, model: str, provider: str, metric: Metric, start_time: datetime,
                                 end_time: datetime, interval: int, stat: Stat):
        with Session(self.engine) as session:
            if metric == Metric.SUCCESS:
                query_result = session.query(ServiceTestResult.input_tokens_count,
                                             ServiceTestResult.output_tokens_count,
                                             ServiceTestResult.time_to_first_token,
                                             ServiceTestResult.total_time,
                                             ServiceTestResult.success,
                                             ServiceTestResult.start_time
                                             ).filter(ServiceTestResult.model_name == model,
                                                      ServiceTestResult.provider_name == provider,
                                                      ServiceTestResult.start_time >= start_time,
                                                      ServiceTestResult.end_time < end_time
                                                      ).order_by(ServiceTestResult.start_time.asc()).all()
            else:
                query_result = session.query(ServiceTestResult.input_tokens_count,
                                             ServiceTestResult.output_tokens_count,
                                             ServiceTestResult.time_to_first_token,
                                             ServiceTestResult.total_time,
                                             ServiceTestResult.success,
                                             ServiceTestResult.start_time
                                             ).filter(ServiceTestResult.model_name == model,
                                                      ServiceTestResult.provider_name == provider,
                                                      ServiceTestResult.start_time >= start_time,
                                                      ServiceTestResult.end_time < end_time,
                                                      ServiceTestResult.success == true()
                                                      ).order_by(ServiceTestResult.start_time.asc()).all()
            rows = list(query_result)
            results = list()
            idx = 0
            interval_start = start_time
            while interval_start < end_time:
                interval_end = interval_start + timedelta(minutes=interval)
                result = {
                    'start_time': interval_start.isoformat(),
                    'end_time': interval_end.isoformat(),
                    'values': list()
                }
                while idx < len(rows) and interval_start <= pytz.UTC.localize(rows[idx][5]) < interval_end:
                    result['values'].append(generate_metric(metric, rows[idx]))
                    idx += 1
                results.append(result)
                interval_start = interval_end
            for result in results:
                result['values'] = generate_stat(metric, stat, result['values'])
            return results

class DatasetQueryEngine:

    def __init__(self, engine: Engine):
        self.engine = engine

    def query_dataset(self, model: str, provider: str, metric: Metric, start_time: datetime,
                                 end_time: datetime, interval: int, stat: Stat):
        with Session(self.engine) as session:
            results = session.query(BenchmarkResult).filter(BenchmarkResult.model_name == model,
            BenchmarkResult.provider_name == provider,
            BenchmarkResult.benchmark == metric,
            BenchmarkResult.success == True
            )

            answer = None
            success = False
            for result in results:
                answer = [{"values":[result.value]}]
                success = True
            
            return answer
