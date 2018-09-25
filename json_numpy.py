# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 18:57:30 2018

@author: Agustin Sanchez Merlinsky
"""

import base64
import json
import numpy as np


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
                return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)