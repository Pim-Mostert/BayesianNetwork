from typing import Dict


class Empty:
    pass


class Cfg(Empty):
    def __init__(self, properties: Dict = None):
        if properties is None:
            return

        for key in properties.keys():
            setattr(self, key, properties[key])

    def ensure(self, name, default_value):
        if not hasattr(self, name):
            setattr(self, name, default_value)
