# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
# ============================================================================
"""Top-level package initialization file.

Modules:
    agents
    catalog
    datasets
    distributions
    generator
    keras
    mplot
    preprocess
    trials
    utils
    visualize
"""

import psiz.agents
import psiz.catalog
import psiz.datasets
import psiz.distributions
import psiz.generators
import psiz.keras
import psiz.mplot
import psiz.preprocess
import psiz.restart
import psiz.trials
import psiz.utils
import psiz.visualize

psiz.models = psiz.keras.models  # TEMPORARY For backwards compatability.
