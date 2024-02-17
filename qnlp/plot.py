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

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(axs, color, label, trainer, model, test_circuits, test_labels):
    ((ax_tl, ax_tr), (ax_bl, ax_br)) = axs
    ax_tl.set_title("Training set")
    ax_tr.set_title("Development set")
    ax_bl.set_xlabel("Epochs")
    ax_br.set_xlabel("Epochs")
    ax_bl.set_ylabel("Accuracy")
    ax_tl.set_ylabel("Loss")

    range_ = np.arange(1, trainer.epochs + 1)
    ax_tl.plot(range_, trainer.train_epoch_costs, color=color, label=label)
    ax_bl.plot(range_, trainer.train_eval_results["acc"], color=color, label=label)
    ax_tr.plot(range_, trainer.val_costs, color=color, label=label)
    ax_br.plot(range_, trainer.val_eval_results["acc"], color=color, label=label)

    ax_tl.legend()
    ax_bl.legend()
    ax_tr.legend()
    ax_br.legend()
