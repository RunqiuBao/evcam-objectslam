import json

class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, tuple) or isinstance(obj, list):
            if isinstance(obj, tuple):
                obj = list(obj)
            for indexItem, itemInObj in enumerate(obj):
                if isinstance(itemInObj, np.ndarray):
                    obj[indexItem] = itemInObj.tolist()
            return obj
        return super(MyJsonEncoder, self).default(obj)


class MyJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def _ConvertListToArrayRecursive(self, obj):
        import numpy as np
        # assume in each level of obj, elements are uniform.
        if isinstance(obj[0], list) and isinstance(obj[0][0], list):
            for indexItem, itemInObj in enumerate(obj):
                obj[indexItem] = self._ConvertListToArrayRecursive(itemInObj)
            return np.array(obj)
        elif isinstance(obj[0], list) and not isinstance(obj[0][0], list):
            for indexItem, itemInObj in enumerate(obj):
                obj[indexItem] = np.array(itemInObj)
            return np.array(obj)
        elif not isinstance(obj[0], list):
            return np.array(obj)
        raise Exception

    def object_hook(self, obj):
        # obj mush be dict
        for key, value in obj.items():
            if isinstance(value, tuple) or isinstance(value, list):
                if isinstance(value, tuple):
                    value = list(value)
                value = self._ConvertListToArrayRecursive(value)
                obj[key] = value
        return obj