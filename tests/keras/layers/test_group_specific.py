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
"""Test GroupSpecific."""

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import tensorflow_probability as tfp

from psiz.keras.layers import EmbeddingNormalDiag
from psiz.keras.layers.group_specific import GroupSpecific


@pytest.fixture
def subnets_determ():
    """A list of subnets"""
    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array)
    )
    emb_1 = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array + 100)
    )
    emb_2 = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array + 200)
    )
    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def subnets_dist_rank0():
    """A list of subnets"""
    # Settings.
    prior_scale = .0000000001
    n_stimuli = 12
    n_dim = 3
    n_sample = ()

    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_0.n_sample = n_sample

    emb_1 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 100
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_1.n_sample = n_sample

    emb_2 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 200
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_2.n_sample = n_sample

    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def subnets_dist_rank1():
    """A list of subnets"""
    # Settings.
    prior_scale = .0000000001
    n_stimuli = 12
    n_dim = 3
    n_sample = 11

    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_0.n_sample = n_sample

    emb_1 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 100
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_1.n_sample = n_sample

    emb_2 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 200
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    emb_2.n_sample = n_sample

    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def group_v0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 1, 1],
                [0, 2, 1]
            ], dtype=np.int32
        )
    )
    return group


@pytest.fixture
def inputs_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [9, 10, 11]
            ], dtype=np.int32
        )
    )
    return inputs


@pytest.fixture
def inputs_v1():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[6, 7, 8], [9, 10, 11]]
            ], dtype=np.int32
        )
    )
    return inputs


def test_subnet_method(subnets_determ):
    group_layer = GroupSpecific(subnets=subnets_determ, group_col=1)
    group_layer.build([[None, None], [None, None]])

    subnet_0 = group_layer.subnet(0)
    subnet_1 = group_layer.subnet(1)
    subnet_2 = group_layer.subnet(2)

    tf.debugging.assert_equal(
        subnet_0.embeddings, subnets_determ[0].embeddings
    )
    tf.debugging.assert_equal(
        subnet_1.embeddings, subnets_determ[1].embeddings
    )
    tf.debugging.assert_equal(
        subnet_2.embeddings, subnets_determ[2].embeddings
    )


def test_call_determ_2d_input(subnets_determ, inputs_v0, group_v0):
    """Test call that does not require an internal reshape."""
    group_layer = GroupSpecific(subnets=subnets_determ, group_col=1)

    outputs = group_layer([inputs_v0, group_v0])

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)


def test_call_determ_3d_input(subnets_determ, inputs_v1, group_v0):
    """Test call with data inputs larger than 2D."""
    group_layer = GroupSpecific(subnets=subnets_determ, group_col=1)

    outputs = group_layer([inputs_v1, group_v0])

    desired_outputs = np.array([
        [
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            [[3.0, 3.1, 3.2], [4.0, 4.1, 4.2], [5.0, 5.1, 5.2]]
        ],
        [
            [
                [100.0, 100.1, 100.2],
                [101.0, 101.1, 101.2],
                [102.0, 102.1, 102.2]
            ],
            [
                [103.0, 103.1, 103.2],
                [104.0, 104.1, 104.2],
                [105.0, 105.1, 105.2]]
        ],
        [
            [
                [200.0, 200.1, 200.2],
                [201.0, 201.1, 201.2],
                [202.0, 202.1, 202.2]
            ],
            [
                [203.0, 203.1, 203.2],
                [204.0, 204.1, 204.2],
                [205.0, 205.1, 205.2]
            ]
        ],
        [
            [
                [106.0, 106.1, 106.2],
                [107.0, 107.1, 107.2],
                [108.0, 108.1, 108.2]
            ],
            [
                [109.0, 109.1, 109.2],
                [110.0, 110.1, 110.2],
                [111.0, 111.1, 111.2]
            ]
        ],
        [
            [
                [206.0, 206.1, 206.2],
                [207.0, 207.1, 207.2],
                [208.0, 208.1, 208.2]
            ],
            [
                [209.0, 209.1, 209.2],
                [210.0, 210.1, 210.2],
                [211.0, 211.1, 211.2]
            ]
        ]
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)


def test_call_dist_2d_input_rank0(subnets_dist_rank0, inputs_v0, group_v0):
    """Test call that does not require an internal reshape."""
    group_layer = GroupSpecific(subnets=subnets_dist_rank0, group_col=1)

    outputs = group_layer([inputs_v0, group_v0])

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        outputs.numpy(), desired_outputs, decimal=1
    )


def test_call_dist_2d_input_rank1(subnets_dist_rank1, inputs_v0, group_v0):
    """Test call that does not require an internal reshape."""
    group_layer = GroupSpecific(subnets=subnets_dist_rank1, group_col=1)

    outputs = group_layer([inputs_v0, group_v0])
    outputs_avg = tf.reduce_mean(outputs, axis=0)

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        outputs_avg.numpy(), desired_outputs, decimal=1
    )


def test_serialization(subnets_dist_rank1, inputs_v0, group_v0):
    # Build group-specific layer.
    group_layer = GroupSpecific(subnets=subnets_dist_rank1, group_col=1)

    # Get configuration.
    config = group_layer.get_config()

    # Reconstruct layer from configuration.
    group_layer_recon = GroupSpecific.from_config(config)

    # Must set `n_sample` since this is not saved.
    for subnet in group_layer_recon.subnets:
        subnet.n_sample = 11

    # Assert reconstructed attributes are the same as original.
    assert group_layer.n_subnet == group_layer_recon.n_subnet
    assert group_layer.group_col == group_layer_recon.group_col

    outputs = group_layer([inputs_v0, group_v0])
    outputs_avg = tf.reduce_mean(outputs, axis=0)

    outputs_recon = group_layer_recon([inputs_v0, group_v0])
    outputs_avg_recon = tf.reduce_mean(outputs_recon, axis=0)

    # Assert calls are roughly equal (given constant initializer).
    np.testing.assert_array_almost_equal(
        outputs_avg.numpy(), outputs_avg_recon.numpy(), decimal=1
    )
