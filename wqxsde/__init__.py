# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os

#import sde_tools
#import wqpget
import wqxsde
from wqxsde.sde_tools import *
from wqxsde.wqpget import *


__version__ = '0.0.1'
__author__ = 'Paul Inkenbrandt'
__name__ = 'wqxsde'

__all__ = ['edit_table','compare_sde_wqx','ProcessStateLabText','table_to_pandas_dataframe','get_field_names']