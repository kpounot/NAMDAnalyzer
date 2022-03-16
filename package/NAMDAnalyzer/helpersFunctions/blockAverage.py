"""A module for block averaging to determine minimum sampling required.

"""


import numpy as np


def blockAverageArray(arr, blocks, axis=0, func=None):
    """Performs a block averaging on an array of values.

    The principle of block averaging is described by Grossfield
    and Tuckerman [#]_.

    Parameters
    ----------
    arr : np.ndarray
        The array to be used for block averaging (e.g. angles,
        rmsd, positions)
    blocks : list, np.array
        A list/array of block sizes.
    axis : int, optional
        The axis on which the observable is computed.
        (default, 0)
    func : callable, optional
        A function to be applied on each block (e.g. calculation
        of mean-square displacement). If None, an average
        is computed on the given *axis* before computing
        the average and standard deviation of the blocks.
        (default, None)

    Returns
    -------
    average : np.ndarray
        The average of the blocks for each block size.
    std : np.ndarray
        The standard deviation for each block size.

    References
    ----------
    .. [#] https://doi.org/10.1016/S1574-1400(09)00502-7

    """
    axisSize = arr.shape[axis]
    outAvg = []
    outStd = []
    for bSize in blocks:
        bList = []
        for bStart in np.arange(0, axisSize - bSize + 1, bSize):
            block = arr.take(range(bStart, bStart + bSize), axis).copy()
            if func is not None:
                block = func(block)
            else:
                block = np.mean(block, axis)
            bList.append(block)
        outAvg.append(np.mean(bList, 0))
        outStd.append(np.std(bList, 0) / np.sqrt(len(bList)))

    return np.array(outAvg), np.array(outStd)


def blockAverageFunc(func, axisSize, blocks):
    """Performs a block averaging on the result of a function call.

    The principle of block averaging is described by Grossfield
    and Tuckerman [#]_.

    Parameters
    ----------
    func : callable
        A function to be called that returns an array or a value
        to perform block averaging on.
        The function should have *bStart* and *bSize* to define the 
        block to be used on the trajectory.
        This could be of the form:

        .. code-block:: python

            lambda bStart, bSize: data.getMSDPerResidue(
                'name CA', ..., frames=slice(bStart, bSize)
            )
    axisSize : int
        The maximum size of the axis on which the block averaging
        is to be performed.
    blocks : list, np.array
        A list/array of block sizes.

    Returns
    -------
    average : np.ndarray
        The average of the blocks for each block size.
    std : np.ndarray
        The standard deviation for each block size.

    References
    ----------
    .. [#] https://doi.org/10.1016/S1574-1400(09)00502-7

    """
    outAvg = []
    outStd = []
    for bSize in blocks:
        bList = []
        for bStart in np.arange(0, axisSize - bSize + 1, bSize):
            block = func(bStart=bStart, bSize=bSize)
            bList.append(block)
        outAvg.append(np.mean(bList, 0))
        outStd.append(np.std(bList, 0) / np.sqrt(len(bList)))

    return np.array(outAvg), np.array(outStd)
