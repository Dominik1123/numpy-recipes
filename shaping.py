import numpy as np


def pad_max_shape(arrays, before=None, after=1, value=0, tie_break=np.floor):
    """Pad the given arrays with a constant values such that their new shapes fit the biggest array.

    Parameters
    ----------
    arrays : sequence of arrays of the same rank
    before, after : {float, sequence, array_like}
        Similar to `np.pad -> pad_width` but specifies the fraction of values to be padded before
        and after respectively for each of the arrays.  Must be between 0 and 1.
        If `before` is given then `after` is ignored.
    value : scalar
        The pad value.
    tie_break : ufunc
        The actual number of items to be padded _before_ is computed as the total number of elements
        to be padded times the `before` fraction and the actual number of items to be padded _after_
        is the remainder. This function determines how the fractional part of the `before` pad width
        is treated. The actual `before` pad with is computed as ``tie_break(N * before).astype(int)``
        where ``N`` is the total pad width. By default `tie_break` just takes the `np.floor` (i.e.
        attributing the fraction part to the `after` pad width). The after pad width is computed as
        ``total_pad_width - before_pad_width``.

    Returns
    -------
    padded_arrays : list of arrays

    Notes
    -----
    By default the `before` pad width is computed as the floor of the `before` fraction times the number
    of missing items for each axis. This is done regardless of whether `before` or `after` is provided
    as a function input. For that reason the fractional part of the `before` pad width is attributed
    to the `after` pad width (e.g. if the total pad width is 3 and the left fraction is 0.5 then the
    `before` pad width is 1 and the `after` pad width is 2; in order to f). This behavior can be controlled
    with the `tie_break` parameter.

    Examples
    --------
    >>> arrays = [np.arange(i) for i in [3, 5, 7]]
    >>> for x in pad_max_shape(arrays):
    ...     print(x)
    ...
    [0 1 2 0 0 0 0]
    [0 1 2 3 4 0 0]
    [0 1 2 3 4 5 6]
    >>> 
    >>> for x in pad_max_shape(arrays, before=1, value=9):
    ...     print(x)
    ...
    [9 9 9 9 0 1 2]
    [9 9 0 1 2 3 4]
    [0 1 2 3 4 5 6]
    >>> for x in pad_max_shape(arrays, after=0.5, value=9)[:-1]:
    ...     print(x)
    ...
    [9 9 0 1 2 9 9]
    [9 0 1 2 3 4 9]

    In case there is an odd number of items to be padded before and after the arrays,
    we can control the behavior via the `tie_break` parameter:

    >>> arrays = [np.arange(1, 4), np.arange(8)]  # 5 items to be padded for array #1.
    >>> print(pad_max_shape(arrays, before=0.5)[0])
    [0 0 1 2 3 0 0 0]
    >>> print(pad_max_shape(arrays, before=0.5, tie_break=np.ceil)[0])
    [0 0 0 1 2 3 0 0]
    
    Max-shape padding works with any number of dimensions:

    >>> arrays = [np.zeros((i, i), dtype=int) + i for i in [3, 5, 7]]
    >>> for x in pad_max_shape(arrays):
    ...     print(x)
    ...
    [[3 3 3 0 0 0 0]
     [3 3 3 0 0 0 0]
     [3 3 3 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    [[5 5 5 5 5 0 0]
     [5 5 5 5 5 0 0]
     [5 5 5 5 5 0 0]
     [5 5 5 5 5 0 0]
     [5 5 5 5 5 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    [[7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]
     [7 7 7 7 7 7 7]]
    >>> for x in pad_max_shape(arrays, before=0.5)[:-1]:
    ...     print(x)
    ...
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    [[0 0 0 0 0 0 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 0 0 0 0 0 0]]

    The padding fractions can be specified per array dimension:

    >>> for x in pad_max_shape(arrays, before=(0, 0.5))[:-1]:
    ...     print(x)
    ...
    [[0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    [[0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 5 5 5 5 5 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]

    The padding fractions can also be specified per array and per dimension:

    >>> for x in pad_max_shape(arrays, before=[(0, 0.5), (0.5, 1), (0, 0)])[:-1]:
    ...     print(x)
    ...
    [[0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 3 3 3 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]]
    [[0 0 0 0 0 0 0]
     [0 0 5 5 5 5 5]
     [0 0 5 5 5 5 5]
     [0 0 5 5 5 5 5]
     [0 0 5 5 5 5 5]
     [0 0 5 5 5 5 5]
     [0 0 0 0 0 0 0]]
    """
    shapes = np.array([x.shape for x in arrays])
    if before is not None:
        before = np.zeros_like(shapes) + before
    else:
        before = np.ones_like(shapes) - after
    max_size = shapes.max(axis=0, keepdims=True)
    margin = (max_size - shapes)
    pad_before = tie_break(margin * before.astype(float)).astype(int)
    pad_after = margin - pad_before
    pad = np.stack([pad_before, pad_after], axis=2)
    return [np.pad(x, w, mode='constant', constant_values=value) for x, w in zip(arrays, pad)]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
