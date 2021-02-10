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
"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.

Example output:
    Attention weights:
          Novice | [3.38 3.32 0.49 0.43]
    Intermediate | [2.06 2.18 2.04 2.18]
          Expert | [0.55 0.50 3.40 3.32]

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.95    0.68    0.16
     Interm |   0.64    0.96    0.54
     Expert |   0.16    0.61    0.96

"""

import os

import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution. TODO
tf.config.experimental_run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility. TODO
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run the simulation that infers an embedding for three groups."""
    # Settings.
    n_stimuli = 30
    n_dim = 4
    n_group = 3
    n_restart = 1
    epochs = 3  # 1000  TODO
    n_trial = 2000
    batch_size = 128

    model_true = ground_truth(n_stimuli, n_dim, n_group)

    # Generate a random docket of trials to show each group.
    generator = psiz.generators.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Create virtual agents for each group.
    agent_novice = psiz.agents.RankAgent(model_true, group_id=0)
    agent_interm = psiz.agents.RankAgent(model_true, group_id=1)
    agent_expert = psiz.agents.RankAgent(model_true, group_id=2)

    # Simulate similarity judgments for each group.
    obs_novice = agent_novice.simulate(docket)
    obs_interm = agent_interm.simulate(docket)
    obs_expert = agent_expert.simulate(docket)
    obs = psiz.trials.stack((obs_novice, obs_interm, obs_expert))

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    # Convert to TF dataset.
    ds_obs_train = obs_train.as_dataset().shuffle(
        buffer_size=obs_train.n_trial, reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=False)
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    ds_obs_test = obs_test.as_dataset().batch(
        batch_size, drop_remainder=False
    )

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=15, mode='min', restore_best_weights=True
    )
    callbacks = [early_stop]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    model_inferred = build_model(n_stimuli, n_dim, n_group)

    # Infer embedding with restarts.
    restarter = psiz.restart.Restarter(
        model_inferred, compile_kwargs=compile_kwargs, monitor='val_loss',
        n_restart=n_restart
    )
    restart_record = restarter.fit(
        x=ds_obs_train, validation_data=ds_obs_val, epochs=epochs,
        callbacks=callbacks, verbose=0
    )
    model_inferred = restarter.model

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_truth = (
        model_similarity(model_true, group_idx=[0]),
        model_similarity(model_true, group_idx=[1]),
        model_similarity(model_true, group_idx=[2])
    )

    simmat_inferred = (
        model_similarity_alt(model_inferred, group_idx=[0]),
        model_similarity_alt(model_inferred, group_idx=[1]),
        model_similarity_alt(model_inferred, group_idx=[2])
    )

    r_squared = np.empty((n_group, n_group))
    for i_truth in range(n_group):
        for j_infer in range(n_group):
            rho, _ = pearsonr(simmat_truth[i_truth], simmat_inferred[j_infer])
            r_squared[i_truth, j_infer] = rho**2

    # Display attention weights.
    # Permute inferred dimensions to best match ground truth.
    attention_weight = model_inferred.kernel.attention.embeddings.numpy()
    idx_sorted = np.argsort(-attention_weight[0, :])
    attention_weight = attention_weight[:, idx_sorted]
    group_labels = ["Novice", "Intermediate", "Expert"]
    print("\n    Attention weights:")
    for i_group in range(attention_weight.shape[0]):
        print("    {0:>12} | {1}".format(
            group_labels[i_group],
            np.array2string(
                attention_weight[i_group, :],
                formatter={'float_kind': lambda x: "%.2f" % x})
            )
        )

    # Display comparison results. A good inferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print('\n    Model Comparison (R^2)')
    print('    ================================')
    print('      True  |        Inferred')
    print('            | Novice  Interm  Expert')
    print('    --------+-----------------------')
    print('     Novice | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[0, 0], r_squared[0, 1], r_squared[0, 2]))
    print('     Interm | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[1, 0], r_squared[1, 1], r_squared[1, 2]))
    print('     Expert | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[2, 0], r_squared[2, 1], r_squared[2, 2]))
    print('\n')


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    stimuli = psiz.keras.layers.Stimuli(
        embedding=psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=.17
            )
        )
    )
    kernel = psiz.keras.layers.AttentionKernel(
        group_level=1,
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        attention=psiz.keras.layers.EmbeddingDeterministic(
            n_group, n_dim, mask_zero=False,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array((
                    (1.8, 1.8, .2, .2),
                    (1., 1., 1., 1.),
                    (.2, .2, 1.8, 1.8)
                ))
            )
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)

    return model


def build_model(n_stimuli, n_dim, n_group):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.

    Returns:
        model: A TensorFlow Keras model.

    """
    stimuli = psiz.keras.layers.Stimuli(
        embedding=psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True,
        )
    )

    # Define group-specific kernels.
    kernel_0 = build_kernel()
    kernel_1 = build_kernel()
    kernel_2 = build_kernel()
    kernel_group = psiz.keras.layers.GroupSpecific(
        [kernel_0, kernel_1, kernel_2], group_col=1
    )

    model = psiz.keras.models.RankAlt(stimuli=stimuli, kernel=kernel_group)

    return model


def build_kernel():
    """Build kernel for single group."""
    # TODO
    # scale = n_dim
    # alpha = np.ones((n_dim))
    # kernel = psiz.keras.layers.AttentionKernel(
    #     group_level=1,
    #     attention=psiz.keras.layers.EmbeddingDeterministic(
    #         n_group, n_dim, mask_zero=False,
    #         embeddings_initializer=psiz.keras.initializers.RandomAttention(
    #             alpha, scale
    #         ),
    #         embeddings_constraint=psiz.keras.constraints.NonNegNorm(
    #             scale=n_dim
    #         ),
    #     ),
    #     similarity=psiz.keras.layers.ExponentialSimilarity()
    # )

    similarity = psiz.keras.layers.ExponentialSimilarity(
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )

    # mink = psiz.keras.layers.MinkowskiStochastic(
    #     rho_loc_trainable=False, rho_scale_trainable=False,
    #     w_loc_trainable=True, w_scale_trainable=False
    # )
    mink = psiz.keras.layers.Minkowski(
        # rho_trainable=False,
        # rho_initializer=tf.keras.initializers.Constant(2.)
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


def model_similarity(model, group_idx=[]):
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        model.stimuli.n_stimuli, mask_zero=True, group_idx=group_idx
    )
    simmat = np.squeeze(
        psiz.utils.pairwise_similarity(
            model.stimuli, model.kernel, ds_pairs
        ).numpy()
    )
    return simmat


def model_similarity_alt(model, group_idx=[]):
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        model.stimuli.n_stimuli, mask_zero=True, group_idx=group_idx
    )
    simmat = np.squeeze(
        pairwise_similarity_alt(
            model.stimuli, model.kernel, ds_pairs
        ).numpy()
    )
    return simmat


def pairwise_similarity_alt(stimuli, kernel, ds_pairs):
    """Return the similarity between stimulus pairs.

    Arguments:
        stimuli: A psiz.keras.layers.Stimuli object.
        kernel: A psiz.keras.layers.Kernel object.
        ds_pairs: A TF dataset object that yields a 3-tuple composed
            of stimulus index i, sitmulus index j, and group
            membership indices.

    Returns:
        s: A tf.Tensor of similarities between stimulus i and stimulus
            j (using the requested group-level parameters from the
            stimuli layer and the kernel layer).
            shape=([sample_size,] n_pair)

    Notes:
        The `n_sample` property of the Stimuli layer and Kernel layer
            must agree.

    """
    s = []
    for x_batch in ds_pairs:
        z_0 = stimuli([x_batch[0], x_batch[2]])
        z_1 = stimuli([x_batch[1], x_batch[2]])
        z_01 = tf.stack([z_0, z_1], axis=-1)
        z_01 = tf.squeeze(z_01, axis=0)
        s.append(
            kernel([z_01, x_batch[2]])
        )

    # Concatenate along pairs dimension (i.e., the last dimension).
    s = tf.concat(s, axis=-1)
    return s


if __name__ == "__main__":
    main()