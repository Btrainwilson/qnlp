# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lambeq import AtomicType, SpiderAnsatz
from lambeq.backend.tensor import Dim
from beartype import beartype
from jax import numpy as np


@beartype
def read_data(filename: str):
    """
    Load data from a file using a given reader.
    Inspired by lambeq tutorial.

    Args:
        filename: str
            The path to the file containing the data.
        reader: lambeq.TreeReader
            The reader to use to convert the data into diagrams.
    Returns:
        list
            A list of diagrams.
    """
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1 - t])
            sentences.append(line[1:].strip())
    return labels, sentences


def build_circuits(data: str, reader):
    diagrams = reader.sentences2diagrams(data)

    ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)})

    return [ansatz(diagram) for diagram in diagrams]
