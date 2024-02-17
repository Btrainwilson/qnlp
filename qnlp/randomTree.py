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

from lambeq import Reader
from lambeq.backend.grammar import Diagram, Ty, Word, Cup
import random

n = Ty("n")


class TreeNode:
    def __init__(self, word, width=1, children=[]):
        self.word = word
        self.children = children
        self.width = width
        if len(children) > 2:
            raise ValueError("Node can only have 2 children")

    def add_child(self, child):
        self.children.append(child)

    def compute_width(self):
        if self.children:
            self.width = sum(child.compute_width() for child in self.children)
        return self.width

    def _tree2diagram(self, offset=0):
        if len(self.children) == 2:
            l_center, l_width, l_cups, l_words = self.children[0]._tree2diagram(offset)

            center = offset + l_width + 1

            r_center, r_width, r_cups, r_words = self.children[1]._tree2diagram(
                center + 2
            )

            cups = l_cups + [[l_center, center - 1], [center + 1, r_center]] + r_cups
            words = l_words + [Word(self.word, n.r @ n @ n.l)] + r_words

            return center, l_width + r_width + 3, cups, words

        else:
            return offset, 1, [], [Word(self.word, n)]


class RandomTreeParser(Reader):
    def _nodes2diagram(self, nodes):
        while len(nodes) > 1:
            # Pick random location to merge
            m_i = random.randint(1, len(nodes) - 1)

            # Merge nodes
            new_node = TreeNode("UNIBOX", width=3, children=nodes[m_i - 1 : m_i + 1])
            # Remove old nodes
            nodes = nodes[: m_i - 1] + [new_node] + nodes[m_i + 1 :]

        _, __, cups, words = nodes[0]._tree2diagram()
        return words, cups

    def sentence2diagram(self, sentence, tokenised=False, debug=False):
        line = sentence.split()
        nodes = [TreeNode(word, i) for i, word in enumerate(line)]
        cups, words = self._nodes2diagram(nodes)
        if debug:
            print(cups)
            print(words)

        return Diagram.create_pregroup_diagram(words, [(Cup, c[0], c[1]) for c in cups])
