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

import torch
from beartype import beartype
from lambeq import PytorchModel, Dataset, AtomicType, PytorchTrainer
import random


N = AtomicType.NOUN
S = AtomicType.SENTENCE
BATCH_SIZE = 30
EPOCHS = 50
LEARNING_RATE = 3e-2
SEED = random.randint(0, 100)


@beartype
def train(train_circuits, train_labels, dev_circuits, dev_labels, test_circuits):
    """
    Train a model using a given reader and data.
    Args:
        train_data: list
            A list of training data.
        test_data: list
            A list of testing data.
        reader: lambeq.TreeReader
            The reader to use to convert the data into diagrams.
    Returns:
        None
    """

    all_circuits = train_circuits + dev_circuits + test_circuits
    model = PytorchModel.from_diagrams(all_circuits)

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,  # type: ignore
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions={"acc": accuracy},
        evaluate_on_train=True,
        verbose="text",
        seed=SEED,
    )

    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)

    dev_dataset = Dataset(dev_circuits, dev_labels)

    trainer.fit(train_dataset, dev_dataset, log_interval=5)

    return trainer, model


def accuracy(y_hat, y):
    return (
        torch.sum(torch.eq(torch.round(torch.sigmoid(y_hat)), y)) / len(y) / 2
    )  # half due to double-counting
