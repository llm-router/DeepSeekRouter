import json
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, render_template

from deepseek_router.metric.metric import MetricQueryEngine, DatasetQueryEngine
from deepseek_router.model.model import convert_code_to_metric, convert_code_to_stat, perf_metrics, benchmark_metrics, Stat, \
    initialize_engine
from deepseek_router.utils.utils import convert_currency, get_supported_currency

app = Flask(__name__)
my_engine = initialize_engine()

services_file_path = Path(__file__).parent / "../../data/services.json"
with services_file_path.open("r") as services_file:
    services = json.load(services_file)

models_file_path = Path(__file__).parent / "../../data/models.json"
with models_file_path.open("r") as models_file:
    models = json.load(models_file)

datasets_file_path = Path(__file__).parent / "../../data/datasets.json"
with datasets_file_path.open("r") as datasets_file:
    datasets = json.load(datasets_file)

keys_file_path = Path(__file__).parent / "../../config/keys.json"
with keys_file_path.open("r") as keys_file:
    keys = json.load(keys_file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/api/list_models', methods=['GET'])
def list_models():
    data = {
        'models': list(),
        'status': True
    }
    for model_code, model_md in models.items():
        model = {
            'model_code': model_code,
            'model_name': model_md['name'],
            'model_type': model_md['type']
        }
        data['models'].append(model)
    return jsonify(data)


@app.route('/api/list_providers', methods=['GET'])
def list_providers():
    model_code = request.args.get('model', None)
    currency = request.args.get('currency', None)
    data = {
        'providers': list()
    }
    err = ''
    if model_code is None:
        err += 'model is a required field. '
    if model_code not in models:
        err += 'model is invalid. '
    if currency is None:
        err += 'currency is a required field. '
    if currency not in get_supported_currency():
        err += 'currency is invalid. '
    if len(err) > 0:
        data['error'] = err
        data['status'] = False
        return jsonify(data)
    data['status'] = True
    for provider_code, provider_md in services[model_code].items():
        if provider_code not in keys or len(keys[provider_code]['default']) == 0:
            continue
        provider = {
            'provider_code': provider_code,
            'provider_name': provider_md['name'],
            'provider_name_cn': provider_md['name_cn'] if 'name_cn' in provider_md else None,
            'provider_homepage': provider_md['homepage'],
            'price_per_million_input_tokens': provider_md['input_price_per_million_token']
            if 'input_price_per_million_token' in provider_md else None,
            'price_per_million_output_tokens': provider_md['output_price_per_million_token']
            if 'output_price_per_million_token' in provider_md else None,
            'price_per_million_cached_input_tokens': provider_md['cached_input_price_per_million_token']
            if 'cached_input_price_per_million_token' in provider_md else None
        }
        provider['price_per_million_input_tokens'] = convert_currency(provider['price_per_million_input_tokens'],
                                                                      provider_md['currency'], currency)
        provider['price_per_million_output_tokens'] = convert_currency(provider['price_per_million_output_tokens'],
                                                                       provider_md['currency'], currency)
        provider['price_per_million_cached_input_tokens'] = convert_currency(
            provider['price_per_million_cached_input_tokens'],
            provider_md['currency'], currency)
        provider = {key: value for key, value in provider.items() if value is not None}
        data['providers'].append(provider)
    return jsonify(data)


@app.route('/api/query_metric', methods=['GET'])
def query_metric():
    data = dict()
    model_code = request.args.get('model', None)
    provider_code = request.args.get('provider', None)
    metric_code = request.args.get('metric', None)
    start_time = request.args.get('start_time', None)
    end_time = request.args.get('end_time', None)
    interval = request.args.get('interval', None)
    stat_code = request.args.get('stat', None)
    now = datetime.now()

    err = ''
    if model_code is None:
        err += 'model is a required field. '
    if model_code not in models:
        err += 'model is invalid. '
    if provider_code is None:
        err += 'provider is a required field. '
    if provider_code not in services[model_code]:
        err += 'provider is invalid. '
    if metric_code is None:
        err += 'metric is a required field. '
    metric = convert_code_to_metric(metric_code)
    if metric is None:
        err += 'metric is invalid. '
    if start_time is not None:
        try:
            start_time = datetime.fromisoformat(start_time)
        except ValueError:
            err += 'start_time is invalid. '
    else:
        start_time = now - timedelta(days=7)
    if end_time is not None:
        try:
            end_time = datetime.fromisoformat(end_time)
        except ValueError:
            err += 'end_time is invalid. '
    else:
        end_time = now
    if interval is not None:
        try:
            interval = int(interval)
        except ValueError:
            err += 'interval is invalid. '
    else:
        interval = 60
    if stat_code is not None:
        stat = convert_code_to_stat(stat_code)
        if stat is None:
            err += 'stat is invalid. '
    else:
        if metric in perf_metrics:
            stat = Stat.Avg
        else:
            stat = Stat.Raw
    if len(err) > 0:
        data['error'] = err
        data['status'] = False
        return jsonify(data)

    if metric in perf_metrics:
        metric_engine = MetricQueryEngine(my_engine)
        result = metric_engine.query_performance_metric(model_code, provider_code, metric,
                                                        start_time, end_time, interval, stat)
        data['status'] = True
        data['datapoints'] = result

    elif metric in benchmark_metrics:
        dataset_engine = DatasetQueryEngine(my_engine)
        result = dataset_engine.query_dataset(model_code, provider_code, metric_code, start_time, end_time, interval,
                                              stat)
        data['status'] = result is not None
        data['datapoints'] = result

    return jsonify(data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
