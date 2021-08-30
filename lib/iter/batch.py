from itertools import chain, islice
from typing import Generator, Iterable, TypeVar

T = TypeVar("T")


def batch(iterable: Iterable[T], n: int) -> Generator[Iterable[T]]:

    if not isinstance(n, int):
        raise TypeError("n must be of type int")

    iterator = iter(iterable)
    for item in iterator:
        yield chain([item], islice(iterator, n - 1))
