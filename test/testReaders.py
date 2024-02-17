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

import os
import qnlp
import lambeq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

test_labels, test_data = qnlp.data.read_data("./test/datasets/mc_test_data.txt")


readers = {
    "random": qnlp.readers.RandomTreeReader(),
    "lambeqtree": lambeq.TreeReader(),
    "detertree": qnlp.readers.DeterministicTreeReader(),
}

for key, reader in readers.items():
    for i in range(3):
        diagram = reader.sentence2diagram(test_data[0])
        diagram.draw(path=f"./test/output/{key}_{i}.pdf", figsize=(10, 10))
