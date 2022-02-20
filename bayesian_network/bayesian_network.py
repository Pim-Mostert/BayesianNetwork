from common.empty import Empty


class BayesianNetwork:
    def __init__(self, cfg: Empty):
        self.cfg = cfg

    def calculate(self):
        return self.cfg.a + self.cfg.b
