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


readers = {
    "bobcat": lambeq.BobcatParser(),
    "treereader": lambeq.TreeReader(),
    "spider": lambeq.spiders_reader,
    "cups": lambeq.cups_reader,
    "stairs": lambeq.stairs_reader,
}

sentence = "woman prepares tasty dinner ."
for i in range(5):
    reader = qnlp.readers.RandomTreeParser()

    d = reader.sentence2diagram(sentence, debug=True)
    d.draw(path=f"./test/output/randomParser_{i}.pdf", figsize=(10, 5))

train_labels, train_data = qnlp.data.read_data("./test/datasets/mc_train_data.txt")
dev_labels, dev_data = qnlp.data.read_data("./test/datasets/mc_dev_data.txt")
test_labels, test_data = qnlp.data.read_data("./test/datasets/mc_test_data.txt")

for key, reader in readers.items():
    print("-" * 40)
    print(key)
    try:
        diagrams = reader.sentences2diagrams(test_data)

        test_circuits = qnlp.data.build_circuits(test_data, reader)
        train_circuits = qnlp.data.build_circuits(train_data, reader)
        dev_circuits = qnlp.data.build_circuits(dev_data, reader)

        trainer, model = qnlp.train_model(
            train_circuits, train_labels, dev_circuits, dev_labels, test_circuits
        )
    except Exception as e:
        print(e)
    print("-" * 40)
