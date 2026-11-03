from __future__ import annotations
from typing import List
from mesh_eval.domain import StandardSampleRecord

class BaseGTAdapter:
    adapter_name = 'gt-base'
    dataset_name = 'unknown'

    def __init__(self, config: object) -> None:
        self.config = config

    def get_samples(self) -> List[StandardSampleRecord]:
        raise NotImplementedError
