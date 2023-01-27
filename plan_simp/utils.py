from typing import Callable, Iterable, List

def lmap(f: Callable, x: Iterable) -> List:
    return list(map(f, x))