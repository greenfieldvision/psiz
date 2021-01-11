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
"""Module for testing models.py."""


import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
# TODO convert to appropriate module once TF updates their docs.
# from tensorflow.compat.v2.test import TestCase

import psiz


@pytest.fixture(scope="module")
def ds_rank_docket():
    """Rank docket dataset."""
    stimulus_set = np.array((
        (0, 1, 2, -1, -1, -1, -1, -1, -1),
        (9, 12, 7, -1, -1, -1, -1, -1, -1),
        (3, 4, 5, 6, 7, -1, -1, -1, -1),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    docket = psiz.trials.RankDocket(stimulus_set, n_select=n_select)

    ds_docket = docket.as_dataset(
        np.zeros([n_trial, 1])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rank_obs():
    """Rank observations dataset."""
    stimulus_set = np.array((
        (0, 1, 2, -1, -1, -1, -1, -1, -1),
        (9, 12, 7, -1, -1, -1, -1, -1, -1),
        (3, 4, 5, 6, 7, -1, -1, -1, -1),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))
    group_id = np.array((0, 0, 1, 1), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, group_id=group_id
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rank_1g_vi():
    n_stimuli = 30
    n_dim = 10
    kl_weight = 0.

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
    )

    prior_scale = .2
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli+1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    stimuli = psiz.keras.layers.Stimuli(embedding=embedding_variational)

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.behavior.RankBehavior()
    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_1g_mle():
    """A MLE rank model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.behavior.RankBehavior()

    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def ds_rate_docket():
    """Rate docket dataset."""
    stimulus_set = np.array((
        (0, 1),
        (9, 12),
        (3, 4),
        (3, 17)
    ), dtype=np.int32)

    n_trial = 4
    docket = psiz.trials.RateDocket(stimulus_set)

    ds_docket = docket.as_dataset(
        np.zeros([n_trial, 1])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rate_obs():
    """Rate observations dataset."""
    n_trial = 4
    stimulus_set = np.array((
        (0, 1),
        (9, 12),
        (3, 4),
        (3, 17)
    ), dtype=np.int32)
    rating = np.array([0.1, .4, .8, .9])
    group_id = np.array((0, 0, 1, 1), dtype=np.int32)

    obs = psiz.trials.RateObservations(
        stimulus_set, rating, group_id=group_id
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rate_1g_mle():
    """A MLE rate model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.behavior.RateBehavior()

    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


def test_propogation_properties(rank_1g_vi):
    """Test propogation properties."""
    assert rank_1g_vi.n_sample == 1

    # Set n_sample at model level.
    rank_1g_vi.n_sample = 100
    # Test property propagated to all consituent layers.
    assert rank_1g_vi.n_sample == 100
    assert rank_1g_vi.stimuli.n_sample == 100
    assert rank_1g_vi.kernel.n_sample == 100
    assert rank_1g_vi.behavior.n_sample == 100
    assert rank_1g_vi.stimuli.embedding.n_sample == 100

    assert rank_1g_vi.kl_weight == 0.

    # Set kl_weight at model level.
    rank_1g_vi.kl_weight = .001
    # Test property propagated to all consituent layers.
    assert rank_1g_vi.kl_weight == .001
    assert rank_1g_vi.stimuli.kl_weight == .001
    assert rank_1g_vi.kernel.kl_weight == .001
    assert rank_1g_vi.behavior.kl_weight == .001
    assert rank_1g_vi.stimuli.embedding.kl_weight == .001

    # # Test invalid support.
    # with pytest.raises(ValueError):
    #     emb.rho = .1
    # with pytest.raises(ValueError):
    #     emb.tau = .1
    # with pytest.raises(ValueError):
    #     emb.mu = 0


def test_save_load_rank(rank_1g_mle, tmpdir, ds_rank_docket, ds_rank_obs):
    """Test loading and saving of embedding model."""
    model = rank_1g_mle
    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs, epochs=1)

    # Predict.
    output_0 = model.predict(ds_rank_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    # TODO test with save_traces
    # model.save(fn, overwrite=True)
    model.save(fn, overwrite=True, save_traces=False)
    # Load the saved model.
    # TODO clean up
    # reconstructed_model = psiz.models.load_model(fn, compile=True)
    # reconstructed_model = tf.keras.models.load_model(
    #     fn, custom_objects={'Rank': psiz.models.Rank}
    # )
    reconstructed_model = tf.keras.models.load_model(fn)
    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rank_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rank_obs, epochs=1)

    # TODO test without compile

    # np.testing.assert_array_equal(
    #     loaded_embedding.z,
    #     rank_1g_vi.z
    # )
    # np.testing.assert_array_equal(
    #     loaded_embedding._z["value"],
    #     rank_1g_vi._z["value"]
    # )
    # assert loaded_embedding._z['trainable'] == rank_1g_vi._z['trainable']

    # assert loaded_embedding._theta == rank_1g_vi._theta

    # np.testing.assert_array_equal(
    #     loaded_embedding.w,
    #     rank_1g_vi.w
    # )
    # for param_name in rank_1g_vi._phi:
    #     np.testing.assert_array_equal(
    #         loaded_embedding._phi[param_name]['value'],
    #         rank_1g_vi._phi[param_name]['value']
    #     )
    #     np.testing.assert_array_equal(
    #         loaded_embedding._phi[param_name]['trainable'],
    #         rank_1g_vi._phi[param_name]['trainable']
    #     )


def test_save_load_rate(rate_1g_mle, tmpdir, ds_rate_docket, ds_rate_obs):
    """Test loading and saving of embedding model."""
    model = rate_1g_mle
    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rate_obs, epochs=1)

    # Predict using original model.
    output_0 = model.predict(ds_rate_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    # TODO test with save_traces
    # model.save(fn, overwrite=True)
    model.save(fn, overwrite=True, save_traces=False)

    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rate_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rate_obs, epochs=1)


# def test_inverse_get_and_set():
#     """Test Inverse getters and setters."""
#     common_parameters_set_get(Inverse)

#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = Inverse(n_stimuli, n_dim=n_dim)

#     # Check in sync following initialization.
#     rho_init = emb.rho
#     tau_init = emb.tau
#     mu_init = emb.mu

#     assert rho_init == emb._theta["rho"]["value"]
#     assert tau_init == emb._theta["tau"]["value"]
#     assert mu_init == emb._theta["mu"]["value"]

#     # Check setters.
#     rho_new = rho_init + 1.1
#     emb.rho = rho_new
#     assert emb.rho == rho_new
#     assert emb._theta["rho"]["value"] == rho_new

#     tau_new = tau_init + 1.1
#     emb.tau = tau_new
#     assert emb.tau == tau_new
#     assert emb._theta["tau"]["value"] == tau_new

#     mu_new = mu_init + 1.1
#     emb.mu = mu_new
#     assert emb.mu == mu_new
#     assert emb._theta["mu"]["value"] == mu_new

#     # Test integer input.
#     emb.rho = 2
#     assert isinstance(emb.rho, float)

#     emb.tau = 2
#     assert isinstance(emb.tau, float)

#     emb.mu = 1
#     assert isinstance(emb.mu, float)

#     # Test invalid support.
#     with pytest.raises(ValueError):
#         emb.rho = .1
#     with pytest.raises(ValueError):
#         emb.tau = .1
#     with pytest.raises(ValueError):
#         emb.mu = 0

#     # Test edges of support.
#     emb.rho = 1
#     emb.tau = 1
#     emb.mu = 0.00001


# def test_exponential_get_and_set():
#     """Test Exponential getters and setters."""
#     common_parameters_set_get(Exponential)

#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = Exponential(n_stimuli, n_dim=n_dim)

#     # Check in sync following initialization.
#     rho_init = emb.rho
#     tau_init = emb.tau
#     gamma_init = emb.gamma
#     beta_init = emb.beta

#     assert rho_init == emb._theta["rho"]["value"]
#     assert tau_init == emb._theta["tau"]["value"]
#     assert gamma_init == emb._theta["gamma"]["value"]
#     assert beta_init == emb._theta["beta"]["value"]

#     # Check setters.
#     rho_new = rho_init + 1.1
#     emb.rho = rho_new
#     assert emb.rho == rho_new
#     assert emb._theta["rho"]["value"] == rho_new

#     tau_new = tau_init + 1.1
#     emb.tau = tau_new
#     assert emb.tau == tau_new
#     assert emb._theta["tau"]["value"] == tau_new

#     gamma_new = gamma_init + .1
#     emb.gamma = gamma_new
#     assert emb.gamma == gamma_new
#     assert emb._theta["gamma"]["value"] == gamma_new

#     beta_new = beta_init + .1
#     emb.beta = beta_new
#     assert emb.beta == beta_new
#     assert emb._theta["beta"]["value"] == beta_new

#     # Test integer input.
#     emb.rho = 2
#     assert isinstance(emb.rho, float)

#     emb.tau = 2
#     assert isinstance(emb.tau, float)

#     emb.gamma = 1
#     assert isinstance(emb.gamma, float)

#     emb.beta = 6
#     assert isinstance(emb.beta, float)

#     # Test invalid support.
#     with pytest.raises(ValueError):
#         emb.rho = .1
#     with pytest.raises(ValueError):
#         emb.tau = .1
#     with pytest.raises(ValueError):
#         emb.gamma = -.1
#     with pytest.raises(ValueError):
#         emb.beta = .1

#     # Test edges of support.
#     emb.rho = 1
#     emb.tau = 1
#     emb.gamma = 0
#     emb.beta = 1


# def test_heavytailed_get_and_set():
#     """Test HeavyTailed getters and setters."""
#     common_parameters_set_get(HeavyTailed)

#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = HeavyTailed(n_stimuli, n_dim=n_dim)

#     # Check in sync following initialization.
#     rho_init = emb.rho
#     tau_init = emb.tau
#     kappa_init = emb.kappa
#     alpha_init = emb.alpha

#     assert rho_init == emb._theta["rho"]["value"]
#     assert tau_init == emb._theta["tau"]["value"]
#     assert kappa_init == emb._theta["kappa"]["value"]
#     assert alpha_init == emb._theta["alpha"]["value"]

#     # Check setters.
#     rho_new = rho_init + 1.1
#     emb.rho = rho_new
#     assert emb.rho == rho_new
#     assert emb._theta["rho"]["value"] == rho_new

#     tau_new = tau_init + 1.1
#     emb.tau = tau_new
#     assert emb.tau == tau_new
#     assert emb._theta["tau"]["value"] == tau_new

#     kappa_new = kappa_init + 1.1
#     emb.kappa = kappa_new
#     assert emb.kappa == kappa_new
#     assert emb._theta["kappa"]["value"] == kappa_new

#     alpha_new = alpha_init + 1.1
#     emb.alpha = alpha_new
#     assert emb.alpha == alpha_new
#     assert emb._theta["alpha"]["value"] == alpha_new

#     # rho=dict(value=2., trainable=True, bounds=[1., None]),
#     # tau=dict(value=1., trainable=True, bounds=[1., None]),
#     # kappa=dict(value=2., trainable=True, bounds=[0., None]),
#     # alpha=dict(value=30., trainable=True, bounds=[0., None])

#     # Test integer input.
#     emb.rho = 2
#     assert isinstance(emb.rho, float)

#     emb.tau = 2
#     assert isinstance(emb.tau, float)

#     emb.kappa = 1
#     assert isinstance(emb.kappa, float)

#     emb.alpha = 4
#     assert isinstance(emb.alpha, float)

#     # Test invalid support.
#     with pytest.raises(ValueError):
#         emb.rho = .1
#     with pytest.raises(ValueError):
#         emb.tau = .1
#     with pytest.raises(ValueError):
#         emb.kappa = -.1
#     with pytest.raises(ValueError):
#         emb.alpha = -.1

#     # Test edges of support.
#     emb.rho = 1
#     emb.tau = 1
#     emb.kappa = 0
#     emb.alpha = 0


# def test_studentst_get_and_set():
#     """Test StudentsT getters and setters."""
#     common_parameters_set_get(StudentsT)

#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = StudentsT(n_stimuli, n_dim=n_dim)

#     # Check in sync follow initialization.
#     rho_init = emb.rho
#     tau_init = emb.tau
#     alpha_init = emb.alpha

#     assert rho_init == emb._theta["rho"]["value"]
#     assert tau_init == emb._theta["tau"]["value"]
#     assert alpha_init == emb._theta["alpha"]["value"]

#     # Check setters.
#     rho_new = rho_init + 1.1
#     emb.rho = rho_new
#     assert emb.rho == rho_new
#     assert emb._theta["rho"]["value"] == rho_new

#     tau_new = tau_init + 1.1
#     emb.tau = tau_new
#     assert emb.tau == tau_new
#     assert emb._theta["tau"]["value"] == tau_new

#     alpha_new = alpha_init + 1.1
#     emb.alpha = alpha_new
#     assert emb.alpha == alpha_new
#     assert emb._theta["alpha"]["value"] == alpha_new

#     # rho=dict(value=2., trainable=False, bounds=[1., None]),
#     # tau=dict(value=2., trainable=False, bounds=[1., None]),
#     # alpha=dict(
#     #     value=(self.n_dim - 1.),
#     #     trainable=False,
#     #     bounds=[0.000001, None]
#     # ),

#     # Test integer input.
#     emb.rho = 2
#     assert isinstance(emb.rho, float)

#     emb.tau = 2
#     assert isinstance(emb.tau, float)

#     emb.alpha = 4
#     assert isinstance(emb.alpha, float)

#     # Test invalid support.
#     with pytest.raises(ValueError):
#         emb.rho = .1
#     with pytest.raises(ValueError):
#         emb.tau = .1
#     with pytest.raises(ValueError):
#         emb.alpha = 0

#     # Test edges of support.
#     emb.rho = 1
#     emb.tau = 1
#     emb.alpha = 0.001


# def common_parameters_set_get(model):
#     """Check common parameters."""
#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = model(n_stimuli, n_dim=n_dim)

#     z_init = emb.z
#     w_init = emb.w

#     # Check in sync following initialization.
#     np.testing.assert_array_equal(z_init, emb._z["value"])
#     np.testing.assert_array_equal(w_init, emb._phi["w"]["value"])

#     set_and_check_z(emb)
#     set_and_check_w(emb)

#     # Two groups.
#     n_group = 2
#     emb = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
#     set_and_check_z(emb)
#     set_and_check_w(emb)

#     # Three groups.
#     n_group = 3
#     emb = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
#     set_and_check_z(emb)
#     set_and_check_w(emb)


# def set_and_check_z(emb):
#     """Set and check z."""
#     z_new = np.random.rand(emb.n_stimuli, emb.n_dim)
#     emb.z = z_new
#     np.testing.assert_array_equal(emb.z, z_new)
#     np.testing.assert_array_equal(emb._z["value"], z_new)

#     z_new = np.random.rand(emb.n_stimuli + 1, emb.n_dim)
#     with pytest.raises(Exception):
#         emb.z = z_new

#     z_new = np.random.rand(emb.n_stimuli, emb.n_dim + 1)
#     with pytest.raises(Exception):
#         emb.z = z_new


# def set_and_check_w(emb):
#     """Set and check w."""
#     w_new = random_weights(emb.n_group, emb.n_dim)
#     emb.w = w_new
#     np.testing.assert_array_equal(emb.w, w_new)
#     np.testing.assert_array_equal(emb._phi["w"]["value"], w_new)

#     w_new = random_weights(emb.n_group + 1, emb.n_dim)
#     with pytest.raises(Exception):
#         emb.w = w_new

#     w_new = random_weights(emb.n_group, emb.n_dim + 1)
#     with pytest.raises(Exception):
#         emb.w = w_new


# def test_inverse_trainable():
#     """Test Inverse trainable."""
#     # Test one group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     emb = Inverse(n_stimuli, n_dim=n_dim)

#     trainable_src = dict(
#         z=True,
#         w=np.array([False]),
#         rho=True,
#         tau=True,
#         mu=True
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     # Test two group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     n_group = 2
#     emb = Inverse(n_stimuli, n_dim=n_dim, n_group=n_group)
#     trainable_src = dict(
#         z=True,
#         w=np.array([True, True]),
#         rho=True,
#         tau=True,
#         mu=True
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     common_parameters_trainable(Inverse, n_group=1)
#     common_parameters_trainable(Inverse, n_group=2)
#     common_parameters_trainable(Inverse, n_group=3)

#     # Model specific parameters.
#     spec_trainable = dict(
#         rho=False, tau=False, mu=False
#     )
#     emb.trainable(spec_trainable)
#     assert not emb._theta["rho"]["trainable"]
#     assert not emb._theta["tau"]["trainable"]
#     assert not emb._theta["mu"]["trainable"]

#     # Default settings.
#     emb.trainable('default')
#     assert emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     assert emb._theta["rho"]["trainable"]
#     assert emb._theta["tau"]["trainable"]
#     assert emb._theta["mu"]["trainable"]

#     emb = Inverse(n_stimuli, n_dim=n_dim)
#     emb.trainable({'w': np.array([True])})
#     emb.trainable('default')
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )


# def test_exponential_trainable():
#     """Test Exponential trainable."""
#     # Test one group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     emb = Exponential(n_stimuli, n_dim=n_dim)

#     trainable_src = dict(
#         z=True,
#         w=np.array([False]),
#         rho=True,
#         tau=True,
#         gamma=True,
#         beta=True,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     # Test two group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     n_group = 2
#     emb = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
#     trainable_src = dict(
#         z=True,
#         w=np.array([True, True]),
#         rho=True,
#         tau=True,
#         gamma=True,
#         beta=True,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     common_parameters_trainable(Exponential, n_group=1)
#     common_parameters_trainable(Exponential, n_group=2)
#     common_parameters_trainable(Exponential, n_group=3)

#     # Model specific parameters.
#     spec_trainable = dict(
#         rho=False, tau=False, gamma=False, beta=False
#     )
#     emb.trainable(spec_trainable)
#     assert not emb._theta["rho"]["trainable"]
#     assert not emb._theta["tau"]["trainable"]
#     assert not emb._theta["gamma"]["trainable"]
#     assert not emb._theta["beta"]["trainable"]

#     # Default settings.
#     emb.trainable('default')
#     assert emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     assert emb._theta["rho"]["trainable"]
#     assert emb._theta["tau"]["trainable"]
#     assert emb._theta["gamma"]["trainable"]
#     assert emb._theta["beta"]["trainable"]

#     emb = Exponential(n_stimuli, n_dim=n_dim)
#     emb.trainable({'w': np.array([True])})
#     emb.trainable('default')
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )


# def test_heavytailed_trainable():
#     """Test HeavyTailed trainable."""
#     # Test one group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     emb = HeavyTailed(n_stimuli, n_dim=n_dim)

#     trainable_src = dict(
#         z=True,
#         w=np.array([False]),
#         rho=True,
#         tau=True,
#         kappa=True,
#         alpha=True,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     # Test two group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     n_group = 2
#     emb = HeavyTailed(n_stimuli, n_dim=n_dim, n_group=n_group)
#     trainable_src = dict(
#         z=True,
#         w=np.array([True, True]),
#         rho=True,
#         tau=True,
#         kappa=True,
#         alpha=True,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     common_parameters_trainable(HeavyTailed, n_group=1)
#     common_parameters_trainable(HeavyTailed, n_group=2)
#     common_parameters_trainable(HeavyTailed, n_group=3)

#     # Model specific parameters.
#     spec_trainable = dict(
#         rho=False, tau=False, kappa=False, alpha=False
#     )
#     emb.trainable(spec_trainable)
#     assert not emb._theta["rho"]["trainable"]
#     assert not emb._theta["tau"]["trainable"]
#     assert not emb._theta["kappa"]["trainable"]
#     assert not emb._theta["alpha"]["trainable"]

#     # Default settings.
#     emb.trainable('default')
#     assert emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     assert emb._theta["rho"]["trainable"]
#     assert emb._theta["tau"]["trainable"]
#     assert emb._theta["kappa"]["trainable"]
#     assert emb._theta["alpha"]["trainable"]

#     emb = HeavyTailed(n_stimuli, n_dim=n_dim)
#     emb.trainable({'w': np.array([True])})
#     emb.trainable('default')
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )


# def test_studentst_trainable():
#     """Test StudentsT trainable."""
#     # Test one group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     emb = StudentsT(n_stimuli, n_dim=n_dim)

#     trainable_src = dict(
#         z=True,
#         w=np.array([False]),
#         rho=False,
#         tau=False,
#         alpha=False,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     # Test two group initialization.
#     n_stimuli = 10
#     n_dim = 3
#     n_group = 2
#     emb = StudentsT(n_stimuli, n_dim=n_dim, n_group=n_group)
#     trainable_src = dict(
#         z=True,
#         w=np.array([True, True]),
#         rho=False,
#         tau=False,
#         alpha=False,
#     )
#     trainable_init = emb.trainable()
#     assert_equal_trainable(trainable_init, trainable_src)

#     common_parameters_trainable(StudentsT, n_group=1)
#     common_parameters_trainable(StudentsT, n_group=2)
#     common_parameters_trainable(StudentsT, n_group=3)

#     # Model specific parameters.
#     spec_trainable = dict(
#         rho=True, tau=True, alpha=True
#     )
#     emb.trainable(spec_trainable)
#     assert emb._theta["rho"]["trainable"]
#     assert emb._theta["tau"]["trainable"]
#     assert emb._theta["alpha"]["trainable"]

#     # Default settings.
#     emb.trainable('default')
#     assert emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     assert not emb._theta["rho"]["trainable"]
#     assert not emb._theta["tau"]["trainable"]
#     assert not emb._theta["alpha"]["trainable"]

#     emb = StudentsT(n_stimuli, n_dim=n_dim)
#     emb.trainable({'w': np.array([True])})
#     emb.trainable('default')
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )


# def common_parameters_trainable(model, n_group=1):
#     """Test trainable functionality of common model parameters."""
#     # One group.
#     n_stimuli = 10
#     n_dim = 3
#     emb = model(n_stimuli, n_dim=n_dim, n_group=n_group)

#     # Toggle z.
#     spec_trainable = dict(z=False)
#     emb.trainable(spec_trainable)
#     assert not emb._z["trainable"]

#     spec_returned = emb.trainable()
#     assert spec_returned["z"] == spec_trainable["z"]

#     spec_trainable = dict(z=True)
#     emb.trainable(spec_trainable)
#     assert emb._z["trainable"]

#     # Toggle w.
#     spec_trainable = dict(w=np.zeros([emb.n_group], dtype=bool))
#     emb.trainable(spec_trainable)
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )

#     returned_spec = emb.trainable()
#     np.testing.assert_array_equal(
#         returned_spec["w"], spec_trainable["w"]
#     )

#     spec_trainable = dict(w=np.ones([emb.n_group], dtype=bool))
#     emb.trainable(spec_trainable)
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )

#     # Toggle z and w.
#     spec_trainable = dict(
#         z=False,
#         w=np.zeros([emb.n_group], dtype=bool)
#     )
#     emb.trainable(spec_trainable)
#     assert not emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )

#     # Test freeze.
#     emb.trainable('freeze')
#     assert not emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.zeros([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     for param_name in emb._theta:
#         assert not emb._theta[param_name]["trainable"]

#     # Test thaw.
#     emb.trainable('thaw')
#     assert emb._z["trainable"]
#     np.testing.assert_array_equal(
#         np.ones([emb.n_group], dtype=bool), emb._phi["w"]["trainable"]
#     )
#     for param_name in emb._theta:
#         assert emb._theta[param_name]["trainable"]


# def assert_equal_trainable(a, b):
#     """Assert that two flat dictionaries are equal."""
#     # Check that they have the same keys.
#     assert set(a.keys()) == set(b.keys())

#     # Check values.
#     for param in a:
#         if type(a[param]) is np.ndarray:
#             np.testing.assert_array_equal(a[param], b[param])
#         else:
#             assert a[param] == b[param]


# def test_private_exponential_similarity():
#     """Test exponential similarity function."""
#     n_stimuli = 10
#     n_dim = 3
#     model = Exponential(n_stimuli, n_dim=n_dim)

#     rho = tf.constant(1.9, dtype=tf.float32)
#     tau = tf.constant(2.1, dtype=tf.float32)
#     beta = tf.constant(1.11, dtype=tf.float32)
#     gamma = tf.constant(.001, dtype=tf.float32)
#     sim_params = {'rho': rho, 'tau': tau, 'gamma': gamma, 'beta': beta}

#     attention_weights = tf.constant(1., shape=[2, 3])

#     z_q = tf.constant(
#         np.array((
#             (.11, -.13, .28),
#             (.45, .09, -1.45)
#         )), dtype=tf.float32, name='z_q'
#     )
#     z_ref = tf.constant(
#         np.array((
#             (.203, -.78, .120),
#             (-.105, -.34, -.278)
#         )), dtype=tf.float32, name='z_r'
#     )

#     s_actual = model._tf_similarity(
#         z_q, z_ref, sim_params, attention_weights
#     )
#     s_desired = np.array([0.60972816, 0.10853130])
#     np.testing.assert_allclose(s_actual, s_desired)


# def test_private_exponential_similarity_broadcast():
#     """Test exponential similarity function with multiple references."""
#     n_stimuli = 10
#     n_dim = 3
#     model = Exponential(n_stimuli, n_dim=n_dim)

#     z_q_in = np.array((
#         (.11, -.13, .28),
#         (.45, .09, -1.45)
#     ))
#     z_q_in = np.expand_dims(z_q_in, axis=2)

#     z_ref_0 = np.array((
#         (.203, -.78, .120),
#         (-.105, -.34, -.278)
#     ))
#     z_ref_1 = np.array((
#         (.302, -.87, .021),
#         (-.501, -.43, -.872)
#     ))
#     z_ref_in = np.stack((z_ref_0, z_ref_1), axis=2)

#     attention_weights = tf.constant(1., shape=[2, 3, 1])

#     rho = tf.constant(1.9, dtype=tf.float32)
#     tau = tf.constant(2.1, dtype=tf.float32)
#     beta = tf.constant(1.11, dtype=tf.float32)
#     gamma = tf.constant(.001, dtype=tf.float32)
#     sim_params = {'rho': rho, 'tau': tau, 'gamma': gamma, 'beta': beta}

#     z_q = tf.constant(
#         z_q_in, dtype=tf.float32, name='z_q'
#     )
#     z_ref = tf.constant(
#         z_ref_in, dtype=tf.float32, name='z_ref'
#     )

#     s_actual = model._tf_similarity(
#         z_q, z_ref, sim_params, attention_weights
#     )
#     s_desired = np.array((
#         [0.60972816, 0.48281544],
#         [0.10853130, 0.16589911]
#     ))
#     print('s_actual:', s_actual)
#     print('s_desired:', s_desired)
#     np.testing.assert_allclose(s_actual, s_desired)


# def test_public_exponential_similarity():
#     """Test similarity function."""
#     # Create Exponential model.
#     n_stimuli = 10
#     n_dim = 3
#     model = Exponential(n_stimuli, n_dim=n_dim)
#     model.rho = 1.9
#     model.tau = 2.1
#     model.beta = 1.11
#     model.gamma = .001
#     model.trainable("freeze")

#     z_q = np.array((
#         (.11, -.13, .28),
#         (.45, .09, -1.45)
#     ))
#     z_ref = np.array((
#         (.203, -.78, .120),
#         (-.105, -.34, -.278)
#     ))
#     # TODO test both cases where attention_weights are and are not provided.
#     # attention_weights = np.ones((2, 3))
#     # s_actual = model.similarity(z_q, z_ref, attention=attention_weights)
#     s_actual = model.similarity(z_q, z_ref)
#     s_desired = np.array([0.60972816, 0.10853130])
#     np.testing.assert_allclose(s_actual, s_desired)


# def test_public_exponential_similarity_broadcast():
#     """Test similarity function."""
#     # Create Exponential model.
#     n_stimuli = 10
#     n_dim = 3
#     model = Exponential(n_stimuli, n_dim=n_dim)
#     model.rho = 1.9
#     model.tau = 2.1
#     model.beta = 1.11
#     model.gamma = .001

#     z_q = np.array((
#         (.11, -.13, .28),
#         (.45, .09, -1.45)
#     ))
#     z_q = np.expand_dims(z_q, axis=2)

#     z_ref_0 = np.array((
#         (.203, -.78, .120),
#         (-.105, -.34, -.278)
#     ))
#     z_ref_1 = np.array((
#         (.302, -.87, .021),
#         (-.501, -.43, -.872)
#     ))
#     z_ref = np.stack((z_ref_0, z_ref_1), axis=2)

#     # TODO test both cases where attention_weights are and are not provided.
#     # attention_weights = np.ones((2, 3))
#     # s_actual = model.similarity(z_q, z_ref, attention=attention_weights)
#     s_actual = model.similarity(z_q, z_ref)
#     s_desired = np.array((
#         [0.60972816, 0.48281544],
#         [0.10853130, 0.16589911]
#     ))
#     np.testing.assert_allclose(s_actual, s_desired)


# def test_probability(model_true, docket_0):
#     """Test probability method."""
#     prob = model_true.outcome_probability(docket_0)  # TODO
#     prob_actual = np.sum(prob, axis=1)
#     prob_desired = np.ones((docket_0.n_trial))
#     np.testing.assert_allclose(prob_actual, prob_desired)


# def test_inflate_points_single_sample(
#         model_true_det, docket_0):
#     """Test inflation with z with 1 sample."""
#     n_reference = 4
#     n_select = 2
#     trial_locs = np.logical_and(
#         docket_0.n_reference == n_reference,
#         docket_0.n_select == n_select
#     )

#     z = model_true_det.z
#     (z_q, z_r) = psiz.models._inflate_points(
#         docket_0.stimulus_set[trial_locs], n_reference,
#         np.expand_dims(z, axis=2)
#     )

#     z_q_desired = np.array(
#         [[0.12737487, 1.3211997], [0.55950886, 1.8086979]],
#         dtype=np.float32)
#     z_q_desired = np.expand_dims(z_q_desired, axis=2)
#     z_q_desired = np.expand_dims(z_q_desired, axis=3)
#     np.testing.assert_allclose(z_q, z_q_desired, rtol=1e-6)

#     z_r_desired = np.array(
#         [
#             [
#                 [0.83358091, 0.88011509, 0.21504205, 0.55950886],
#                 [1.52554786, 0.64515489, 0.92544436, 1.80869794]
#             ],
#             [
#                 [1.90893364, 2.8184545, -0.04342473, 0.83358091],
#                 [-0.15246096, 2.63771772, 1.41283584, 1.52554786]
#             ]
#         ], dtype=np.float32)
#     z_r_desired = np.expand_dims(z_r_desired, axis=3)
#     np.testing.assert_allclose(z_r, z_r_desired, rtol=1e-6)


# def test_inflate_points_multiple_samples(model_true_det):
#     """Test inflation when z contains samples."""
#     n_stimuli = 7
#     n_dim = 2
#     n_sample = 10
#     n_reference = 3
#     n_trial = 5
#     stimulus_set = np.array((
#         (0, 1, 2, 3),
#         (3, 4, 0, 1),
#         (3, 5, 6, 1),
#         (1, 4, 5, 6),
#         (2, 1, 3, 6),
#     ))
#     mean = np.zeros((n_dim))
#     cov = .1 * np.identity(n_dim)
#     z = np.random.multivariate_normal(mean, cov, (n_sample, n_stimuli))
#     z = np.transpose(z, axes=[1, 2, 0])

#     # Desired inflation outcome.
#     z_q_desired = z[stimulus_set[:, 0], :, :]
#     z_q_desired = np.expand_dims(z_q_desired, axis=2)
#     z_r_desired = np.empty((n_trial, n_dim, n_reference, n_sample))
#     for i_ref in range(n_reference):
#         z_r_desired[:, :, i_ref, :] = z[stimulus_set[:, 1+i_ref], :]

#     (z_q, z_r) = psiz.models._inflate_points(
#         stimulus_set, n_reference, z)

#     np.testing.assert_allclose(z_q, z_q_desired, rtol=1e-6)
#     np.testing.assert_allclose(z_r, z_r_desired, rtol=1e-6)


# def test_tf_ranked_sequence_probability(model_true, docket_0):
#     """Test tf_ranked_sequence_probability."""
#     docket = docket_0
#     n_reference = 4
#     n_select = 2
#     trial_locs = np.logical_and(
#         docket.n_reference == n_reference,
#         docket.n_select == n_select
#     )

#     tf_config = [
#         tf.constant(1),
#         tf.constant([n_reference]),
#         tf.constant([n_select]),
#         tf.constant([True])
#     ]
#     inner_model = model_true._build_model(tf_config)

#     z = model_true.z

#     attention = model_true._phi["w"]["value"][0, :]
#     attention = np.tile(attention, (docket_0.n_trial, 1))

#     (z_q, z_r) = psiz.models._inflate_points(
#         docket.stimulus_set[trial_locs], n_reference,
#         np.expand_dims(z, axis=2)
#     )
#     s_qr = model_true.similarity(z_q, z_r, group_id=0)
#     prob_1 = psiz.models._ranked_sequence_probability(s_qr, n_select)
#     prob_1 = prob_1[:, 0]

#     # NOTE: tf_ranked_sequence_probability is not implemented to handle
#     # samples.
#     tf_n_select = tf.constant(n_select, dtype=tf.int32)
#     tf_s_qr = tf.constant(s_qr[:, :, 0], dtype=tf.float32)
#     prob_2 = psiz.models._tf_ranked_sequence_probability(
#         tf_s_qr, tf_n_select
#     )
#     np.testing.assert_allclose(prob_1, prob_2, rtol=1e-6)


# def test_tuning_distance():
#     """Test alternative formulation of distance."""
#     z_q = np.array((
#         (.11, -.13, .28),
#         (.45, .09, -1.45),
#         (.21, .14, .58),
#         (-.91, -.41, -.19)
#     ))

#     z_r = np.array((
#         (.20, -.78, .12),
#         (-.10, -.34, -.28),
#         (.03, .38, -.12),
#         (-.15, -.42, -.78)
#     ))

#     rho = 2

#     attention = np.array((
#         (1., 1.2, .8),
#         (1., 1.2, .8),
#         (1., 1.2, .8),
#         (1., 1.2, .8)
#     ))

#     d_qr_0 = (np.abs(z_q - z_r))**rho
#     d_qr_0 = np.multiply(d_qr_0, attention)
#     d_qr_0 = np.sum(d_qr_0, axis=1)**(1. / rho)

#     # Note: The weight matrix is 3D tensor, first dimension corresponds to
#     # the pair being compared.
#     # Note: matmul treats the last two dimensions as the actual matrices and
#     # broadcasts appropriately.

#     # Common weight matrix.
#     w1 = np.array((
#         (1., 0., 0.),
#         (0., 1.2, 0.),
#         (0., 0., .8)
#     ))
#     w = np.expand_dims(w1, axis=0)
#     x = np.expand_dims(np.abs(z_q - z_r)**(rho / 2), axis=2)
#     x_t = np.transpose(x, axes=(0, 2, 1))
#     d_qr_1 = np.matmul(x_t, w)
#     d_qr_1 = np.matmul(d_qr_1, x)
#     d_qr_1 = d_qr_1**(1 / rho)
#     d_qr_1 = np.squeeze(d_qr_1)
#     np.testing.assert_array_almost_equal(d_qr_0, d_qr_1)

#     # Separate weight matrix.
#     attention = np.array((
#         (1., 1.2, .8),
#         (1., 1.2, .8),
#         (.7, 1., 1.3),
#         (.7, 1., 1.3),
#     ))
#     d_qr_0 = (np.abs(z_q - z_r))**rho
#     d_qr_0 = np.multiply(d_qr_0, attention)
#     d_qr_0 = np.sum(d_qr_0, axis=1)**(1. / rho)

#     w1 = np.array((
#         (1., 0., 0.),
#         (0., 1.2, 0.),
#         (0., 0., .8)
#     ))
#     w1 = np.tile(w1, [2, 1, 1])
#     w2 = np.array((
#         (.7, 0., 0.),
#         (0., 1., 0.),
#         (0., 0., 1.3)
#     ))
#     w2 = np.tile(w2, [2, 1, 1])
#     w = np.concatenate((w1, w2), axis=0)
#     x = np.expand_dims(np.abs(z_q - z_r)**(rho / 2), axis=2)
#     x_t = np.transpose(x, axes=(0, 2, 1))
#     d_qr_1 = np.matmul(x_t, w)
#     d_qr_1 = np.matmul(d_qr_1, x)
#     d_qr_1 = d_qr_1**(1 / rho)
#     d_qr_1 = np.squeeze(d_qr_1)
#     np.testing.assert_array_almost_equal(d_qr_0, d_qr_1)


# def test_tuning_distance_with_multiple_references():
#     """Test alternative formulation of distance."""
#     z_q = np.array((
#         (.11, -.13, .28),
#         (.45, .09, -1.45),
#         (.21, .14, .58),
#         (-.91, -.41, -.19)
#     ))
#     z_q = np.expand_dims(z_q, axis=2)

#     z_r = np.array((
#         (.20, -.78, .12),
#         (-.10, -.34, -.28),
#         (.03, .38, -.12),
#         (-.15, -.42, -.78)
#     ))
#     z_r = np.expand_dims(z_r, axis=2)
#     z_r = np.tile(z_r, [1, 1, 2])

#     rho = 2

#     attention = np.array((
#         (1., 1.2, .8),
#         (1., 1.2, .8),
#         (1., 1.2, .8),
#         (1., 1.2, .8)
#     ))
#     attention = np.expand_dims(attention, axis=2)

#     d_qr_0 = (np.abs(z_q - z_r))**rho
#     d_qr_0 = np.multiply(d_qr_0, attention)
#     d_qr_0 = np.sum(d_qr_0, axis=1)**(1. / rho)

#     # Note: The weight matrix is a 3D tensor, first dimension corresponds to
#     # the pair being compared.
#     # Note: matmul treats the last two dimensions as the actual matrices and
#     # broadcasts appropriately.

#     # Common weight matrix.
#     w1 = np.array((
#         (1., 0., 0.),
#         (0., 1.2, 0.),
#         (0., 0., .8)
#     ))
#     w = np.expand_dims(w1, axis=0)
#     w = np.expand_dims(w, axis=0)
#     x = np.abs(z_q - z_r)**(rho / 2)
#     x = np.transpose(x, axes=(0, 2, 1))
#     x = np.expand_dims(x, axis=3)

#     x_t = np.transpose(x, axes=(0, 1, 3, 2))
#     d_qr_1 = np.matmul(x_t, w)
#     d_qr_1 = np.matmul(d_qr_1, x)
#     d_qr_1 = d_qr_1**(1 / rho)
#     d_qr_1 = np.squeeze(d_qr_1)
#     np.testing.assert_array_almost_equal(d_qr_0, d_qr_1)

#     # Separate weight matrix.
#     # attention = np.array((
#     #     (1., 1.2, .8),
#     #     (1., 1.2, .8),
#     #     (.7, 1., 1.3),
#     #     (.7, 1., 1.3),
#     # ))
#     # d_qr_0 = (np.abs(z_q - z_r))**rho
#     # d_qr_0 = np.multiply(d_qr_0, attention)
#     # d_qr_0 = np.sum(d_qr_0, axis=1)**(1. / rho)

#     # w1 = np.array((
#     #     (1., 0., 0.),
#     #     (0., 1.2, 0.),
#     #     (0., 0., .8)
#     # ))
#     # w1 = np.tile(w1, [2, 1, 1])
#     # w2 = np.array((
#     #     (.7, 0., 0.),
#     #     (0., 1., 0.),
#     #     (0., 0., 1.3)
#     # ))
#     # w2 = np.tile(w2, [2, 1, 1])
#     # w = np.concatenate((w1, w2), axis=0)
#     # x = np.expand_dims(np.abs(z_q - z_r)**(rho / 2), axis=2)
#     # x_t = np.transpose(x, axes=(0, 2, 1))
#     # d_qr_1 = np.matmul(x_t, w)
#     # d_qr_1 = np.matmul(d_qr_1, x)
#     # d_qr_1 = d_qr_1**(1 / rho)
#     # d_qr_1 = np.squeeze(d_qr_1)
#     # np.testing.assert_array_almost_equal(d_qr_0, d_qr_1)


# def random_weights(n_group, n_dim):
#     """Generate random attention weights."""
#     w = np.random.rand(n_group, n_dim)
#     w = w / np.sum(w, axis=1, keepdims=True)
#     w = w * n_dim
#     return w


# TODO
# emb_inferred.save('tmp_emb', overwrite=True)
# emb_inferred2 = psiz.models.Proxy(psiz.models.load_model('tmp_emb'))
# simmat_infer = psiz.utils.pairwise_matrix(
#     emb_inferred2.similarity, emb_inferred2.z
# )
# r_squared = psiz.utils.matrix_comparison(
#     simmat_truth, simmat_infer, score='r2'
# )
# np.testing.assert_array_equal(emb_inferred.z, emb_inferred2.z)
# np.testing.assert_array_equal(emb_inferred.w, emb_inferred2.w)

# print(
#     '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
# )


# TODO move elsewhere
# def test_weight_projections():
#     """Test projection of attention weights."""
#     # Create Exponential model.
#     constraint = ProjectAttention()
#     n_dim = 3

#     attention = np.array(
#         (
#             (1., 1., 1.),
#             (2., 1., 1.)
#         ), ndmin=2
#     )
#     tf_attention = tf.constant(
#         attention, dtype=tf.float32
#     )

#     # Project attention weights.
#     total = np.sum(attention, axis=1, keepdims=True)
#     attention_desired = n_dim * attention / total

#     attention_actual = constraint(tf_attention)
#     np.testing.assert_allclose(attention_actual, attention_desired)


# class ProjectAttentionTest(TestCase):
#     """Test ProjectAttention constraint."""

#     def testProjectAttention(self):
#         """Test."""
#         n_dim = 3
#         attention = np.array(
#             (
#                 (1., 1., 1.),
#                 (2., 1., 1.)
#             ), ndmin=2
#         )
#         total = np.sum(attention, axis=1, keepdims=True)
#         attention_desired = n_dim * attention / total

#         tf_attention = tf.convert_to_tensor(
#             attention, dtype=tf.float32
#         )
#         with self.session(use_gpu=False):
#             attention_actual = ProjectAttention()(tf_attention).eval()
#             self.assertEqual(attention_actual, attention_desired)
