from __future__ import annotations
from typing import Any, Dict, Optional
import os
import pickle as pkl

class Cache(dict):
    def __init__(self, cache_init: Optional[Dict]=None) -> None:
        assert cache_init is None or isinstance(cache_init, dict)
        if cache_init is None:
            cache_init = {}
        super().__init__(cache_init)
        # self.cache_hit_rate = 1.0

    def dump(self, file_name: str):
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            pkl.dump(self, f)
    
    def load(self, file_name: str):
        with open(file_name, 'rb') as f:
            self.update(pkl.load(f))
    
    def __getitem__(self, key: str) -> Dict:
        # self.cache_hit_rate = (self.cache_hit_rate * 0.99) + 0.01
        return super().__getitem__(key)
    
    def __setitem__(self, key: str, newvalue: Any):
        # self.cache_hit_rate = self.cache_hit_rate * 0.99
        return super().__setitem__(key, newvalue)
    
    def get_hit_rate(self):
        return self.cache_hit_rate
