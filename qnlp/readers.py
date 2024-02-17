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

import warnings

warnings.filterwarnings("ignore")

from lambeq import AtomicType, Reader
from lambeq.backend.grammar import Box, Diagram, Id, Ty, Word, Cup
import random


N = AtomicType.NOUN
S = AtomicType.SENTENCE

n = Ty("n")


class CombReader(Reader):
    def sentence2diagram(self, sentence):
        words = Id().tensor(*[Word(w, S) for w in sentence.split()])
        layer = Box("LAYER", words.cod, S)
        return words >> layer


class RandomTreeReader(Reader):
    def _tree2diagram(self, line):
        if len(line) == 1:
            words = Word(line[0], n)
            return words

        rand_pos = random.randint(1, len(line) - 1)
        left = self._tree2diagram(line[:rand_pos])
        right = self._tree2diagram(line[rand_pos:])

        words = Id().tensor(*[left, right])
        layer = Box("UNIBOX", words.cod, S)
        return words >> layer

    def sentence2diagram(self, sentence, tokenised=False):
        line = sentence.split()
        d = self._tree2diagram(line)
        return d


class DeterministicTreeReader(Reader):
    def _tree2diagram(self, left, right):
        if len(left) == 1:
            words_l = [Word(left[0], n)]
            l_center = 0
            l_u = 0
            cups_l = []

        else:
            rand_pos = len(left) // 2
            l_center, l_u, cups_l, words_l = self._tree2diagram(
                left[:rand_pos], left[rand_pos:]
            )

        center = l_u + 2

        if len(right) == 1:
            words_r = [Word(right[0], n)]
            r_center = center + 2
            r_u = r_center
            cups_r = []

        else:
            rand_pos = len(right) // 2
            r_center, r_u, cups_r, words_r = self._tree2diagram(
                right[:rand_pos], right[rand_pos:]
            )
            r_u = center + r_u + 2
            r_center = center + r_center + 2

        cups_l.extend([[l_center, center - 1], [center + 1, r_center]])
        cups_r = [[rc[0] + center + 2, rc[1] + center + 2] for rc in cups_r]
        cups_l.extend(cups_r)

        words_l.append(Word("UNIBOX", n.r @ n @ n.l))
        words_l.extend(words_r)
        return center, r_u, cups_l, words_l

    def sentence2diagram(self, sentence, tokenised=False):
        line = sentence.split()
        rand_pos = random.randint(1, len(line) - 1)
        _, __, cups, words = self._tree2diagram(line[:rand_pos], line[rand_pos:])
        return Diagram.create_pregroup_diagram(words, [(Cup, c[0], c[1]) for c in cups])


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
    def sentence2diagram(self, sentence, tokenised=False, debug=False):
        line = sentence.split()
        nodes = [TreeNode(word, i) for i, word in enumerate(line)]

        while len(nodes) > 1:
            # Pick random location to merge
            m_i = random.randint(1, len(nodes) - 1)

            # Merge nodes
            new_node = TreeNode("UNIBOX", width=3, children=nodes[m_i - 1 : m_i + 1])
            # Remove old nodes
            nodes = nodes[: m_i - 1] + [new_node] + nodes[m_i + 1 :]

        _, __, cups, words = nodes[0]._tree2diagram()
        if debug:
            print(cups)
            print(words)
        return Diagram.create_pregroup_diagram(words, [(Cup, c[0], c[1]) for c in cups])
