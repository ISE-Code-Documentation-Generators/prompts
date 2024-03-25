from typing import Any, Callable, Generic, Optional, Sequence, List, TypeVar


_EL = TypeVar("_EL")
_REDUCE = TypeVar("_REDUCE")
_MAP = TypeVar("_MAP")

class Pipeline(Generic[_EL]):
    def __init__(self, l: Sequence[_EL]):
        self.l = l
    
    def to_map(self, f: "Callable[[_EL], _MAP]") -> "Pipeline[_MAP]":
        return Pipeline(list(map(f, self.l)))
    
    def to_reduce(self, f: "Callable[[Optional[_REDUCE], _EL], _REDUCE]", initial: Optional[_REDUCE]=None) -> Optional[_REDUCE]:
        from functools import reduce
        return reduce(f, self.l, initial)
    
    def to_list(self) -> "List[_EL]":
        return list(self.l)