from const import SUCCESS, FAILED


class QuickResponse:
    @staticmethod
    def result(result, **kwargs):
        return {
            'result': result,
            'data': kwargs
        }

    @staticmethod
    def error(exception):
        if isinstance(exception, Exception):
            _error = type(exception).__name__
        else:
            _error = str(exception)
        return {
            'result': FAILED,
            'data': {
                'error': _error
            }
        }

    @staticmethod
    def success():
        _result = SUCCESS
        return {'result': SUCCESS}

    @staticmethod
    def data(**kwargs):
        _data = kwargs
        return {
            'result': SUCCESS,
            'data': kwargs
        }
