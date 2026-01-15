from typing import Protocol, Any

class ArrayLike(Protocol):
    shape: tuple
    dtype: Any
