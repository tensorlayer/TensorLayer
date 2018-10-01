#! /usr/bin/python
# -*- coding: utf-8 -*-

__all__ = ['list_remove_repeat']


def list_remove_repeat(x):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    x : list
        Input

    Returns
    -------
    list
        A list that after removing it's repeated items

    Examples
    -------
    >>> l = [2, 3, 4, 2, 3]
    >>> l = list_remove_repeat(l)
    [2, 3, 4]

    """
    y = []
    for i in x:
        if i not in y:
            y.append(i)

    return y