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
import torch
from lambeq import BobcatParser
import qnlp
import lambeq
import matplotlib.pyplot as plt


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# To test the lambeq readers.
# readers = {"bobcat" : lambeq.BobcatParser(),
#            "treereader" : lambeq.TreeReader(),
#            "spider" : lambeq.spiders_reader,
#            "stairs" : lambeq.stairs_reader}

# colors = {"bobcat": "blue", "treereader": "red", "spider": "green", "stairs": "orange"}

# Working reader
readers = {
    "StaticReader": qnlp.readers.DeterministicTreeReader(),
    "RandomReader": qnlp.readers.RandomTreeParser(),
}

colors = {"StaticReader": "blue", "RandomReader": "red"}
fig1, axs = plt.subplots(2, 2, sharey="row", figsize=(10, 6))
fig1.suptitle("Training and Testing Different Readers")

train_labels, train_data = qnlp.data.read_data("./test/datasets/mc_train_data.txt")
dev_labels, dev_data = qnlp.data.read_data("./test/datasets/mc_dev_data.txt")
test_labels, test_data = qnlp.data.read_data("./test/datasets/mc_test_data.txt")

for key, reader in readers.items():
    diagrams = reader.sentences2diagrams(test_data)

    test_circuits = qnlp.data.build_circuits(test_data, reader)
    train_circuits = qnlp.data.build_circuits(train_data, reader)
    dev_circuits = qnlp.data.build_circuits(dev_data, reader)

    trainer, model = qnlp.train_model(
        train_circuits, train_labels, dev_circuits, dev_labels, test_circuits
    )

    qnlp.plot.plot_accuracy(
        axs, colors[key], key, trainer, model, test_circuits, test_labels
    )

    test_acc = qnlp.train.accuracy(
        model.forward(test_circuits), torch.tensor(test_labels)
    )
    print("Testing Accuracy: ", test_acc.item())

plt.savefig("./test/output/figure.pdf")
