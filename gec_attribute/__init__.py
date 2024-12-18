import gec_attribute.methods as methods
import gec_attribute.metrics as metrics
import inspect

def get_metric_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    metric_ids = [
        elem[0].lower() for elem in inspect.getmembers(metrics, inspect.isclass) \
            if not elem[0].lower().startswith('metricbase')
    ]
    return metric_ids

def get_metric(metric_id: str):
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if not metric_id in get_metric_ids():
        raise ValueError(f'The metric_id should be {get_metric_ids()}. Your input is {metric_id}.')
    metric_dict = {
        elem[0].lower(): elem[1] for elem in inspect.getmembers(metrics, inspect.isclass) \
              if not elem[0].lower().startswith('metricbase')
    }
    return metric_dict[metric_id]

def get_method_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    method_ids = [
        elem[0].lower().replace('attribution', '') for elem in inspect.getmembers(methods, inspect.isclass) \
            if not elem[0].lower().startswith('attributionbase')
    ]
    return method_ids

def get_method(method_id: str):
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if not method_id in get_method_ids():
        raise ValueError(f'The metric_id should be {get_method_ids()}. Your input is {method_id}.')
    method_dict = {
        elem[0].lower().replace('attribution', ''): elem[1] for elem in inspect.getmembers(methods, inspect.isclass) \
              if not elem[0].lower().startswith('attributionbase')
    }
    return method_dict[method_id]