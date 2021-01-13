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
"""Module of custom TensorFlow regularizers.

Classes:
    Squeeze:

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='psiz.keras.regularizers')
class Squeeze(tf.keras.regularizers.Regularizer):
    """Squeeze representation into a low number of dimensions."""

    def __init__(self, rate=0.):
        """Initialize.

        Arguments:
            rate: Rate at which regularization is applied.

        """
        self.rate = rate

    def __call__(self, z):
        """Call."""
        # Sum across stimuli, but within a dimension.
        dimension_usage = tf.reduce_max(tf.math.abs(z), axis=0)
        return self.rate * tf.reduce_sum(dimension_usage)

    def get_config(self):
        """Return config."""
        return {'rate': float(self.rate)}