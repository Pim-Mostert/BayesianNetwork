class Empty:
    pass


class Cfg(Empty):
    def ensure(self, name, default_value):
        if not hasattr(self, name):
            setattr(self, name, default_value)
