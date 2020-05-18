import numpy as np
import nni


def run_trial(params):
    res = np.random.randn(1).item() * params['a'] + params['b']
    param1 = params['a']
    param2 = params['b']
    nni.report_intermediate_result({'name': 'param1', 'default': param1})
    nni.report_intermediate_result({'name': 'param2', 'default': param2})
    nni.report_final_result(res)


if __name__ == '__main__':
    params = nni.get_next_parameter()
    run_trial(params)
