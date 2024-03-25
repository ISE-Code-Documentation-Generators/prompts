class Pipeline:
    def __init__(self, l):
        self.l = l
    
    def to_map(self, f):
        return Pipeline(list(map(f, self.l)))
    
    def to_reduce(self, f, initial=None):
        from functools import reduce
        return reduce(f, self.l, initial)
    
    def to_list(self):
        return self.l