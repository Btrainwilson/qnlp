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

from lambeq.backend.grammar import Diagram, Ty, Word, Cup


# Testing how to use the lambeq library for generating diagrams using cups

n, s = Ty("n"), Ty("s")

words = [
    Word("my", n),
    Word("heart", n.r @ n @ n.l),
    Word("will", n),
    Word("go", n.r @ n @ n.l),
    Word("on", n),
]
morphs = [
    (Cup, 0, 1),
    (Cup, 3, 4),
    (Cup, 2, 5),
    (Cup, 7, 8),
]
diagram = Diagram.create_pregroup_diagram(words, morphs)
diagram.draw()
