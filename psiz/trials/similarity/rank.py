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
"""Rank trials module.

On each similarity judgment trial, an agent judges the similarity
between a single query stimulus and multiple reference stimuli.

Classes:
    RankTrials: Abstract base class for 'Rank' trials.
    RankDocket: Unjudged 'Rank' trials.
    RankObservations: Judged 'Rank' trials.

"""

from abc import ABCMeta, abstractmethod
from itertools import permutations
import copy
import warnings

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.similarity.base import SimilarityTrials
from psiz.utils import pad_2d_array


class RankTrials(SimilarityTrials, metaclass=ABCMeta):
    """Abstract base class for rank-type trials."""

    def __init__(self, stimulus_set, n_select=None, is_ranked=None):
        """Initialize.

        Arguments:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. The first
                column is the query stimulus. The remaining columns
                indicate reference stimuli. It is assumed that stimuli
                indices are composed of integers from [0, N-1], where N
                is the number of unique stimuli. The value -1 can be
                used as a placeholder for non-existent references.
                shape = (n_trial, max(n_reference) + 1)
            n_select (optional): An integer array indicating the number
                of references selected in each trial. Values must be
                greater than zero but less than the number of
                references for the corresponding trial.
                shape = n_trial,)
            is_ranked (optional): A Boolean array indicating which
                trials require reference selections to be ranked.
                shape = (n_trial,)

        """
        SimilarityTrials.__init__(self, stimulus_set)

        n_reference = self._infer_n_reference(stimulus_set)
        self.n_reference = self._check_n_reference(n_reference)

        # Format stimulus set.
        self.max_n_reference = np.amax(self.n_reference)
        self.stimulus_set = self.stimulus_set[:, 0:self.max_n_reference+1]

        if n_select is None:
            n_select = np.ones((self.n_trial), dtype=np.int32)
        else:
            n_select = self._check_n_select(n_select)
        self.n_select = n_select

        if is_ranked is None:
            is_ranked = np.full((self.n_trial), True)
        else:
            is_ranked = self._check_is_ranked(is_ranked)
        self.is_ranked = is_ranked

    def _infer_n_reference(self, stimulus_set):
        """Return the number of references in each trial.

        Infers the number of available references for each trial. The
        function assumes that values less than zero, are placeholder
        values and should be treated as non-existent.

        Arguments:
            stimulus_set: shape = [n_trial, 1]

        Returns:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = [n_trial, 1]

        """
        n_reference = self._infer_n_present(stimulus_set) - 1
        return n_reference.astype(dtype=np.int32)

    def _check_n_reference(self, n_reference):
        """Check the argument `n_reference`.

        Raises:
            ValueError

        """
        if np.sum(np.less(n_reference, 2)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least three "
                "non-negative integers per a row, i.e. one query and at least "
                "two reference stimuli per trial."))
        return n_reference

    def _check_n_select(self, n_select):
        """Check the argument `n_select`.

        Raises:
            ValueError

        """
        n_select = n_select.astype(np.int32)
        # Check shape agreement.
        if not (n_select.shape[0] == self.n_trial):
            raise ValueError((
                "The argument `n_select` must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = n_select < 1
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The argument `n_select` contains integers less than 1. "
                "Found {0} bad trial(s).").format(n_bad))
        # Check upperbound support limit.
        bad_locs = np.greater_equal(n_select, self.n_reference)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The argument `n_select` contains integers greater than "
                "or equal to the corresponding 'n_reference'. Found {0} bad "
                "trial(s).").format(n_bad))
        return n_select

    def _check_is_ranked(self, is_ranked):
        """Check the argument `is_ranked`.

        Raises:
            ValueError

        """
        if not (is_ranked.shape[0] == self.n_trial):
            raise ValueError((
                "The argument `n_select` must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        bad_locs = np.not_equal(is_ranked, True)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The unranked version is not implemented, Found {0} bad "
                "trial(s).").format(n_bad))
        return is_ranked

    def is_select(self, compress=False):
        """Indicate if a stimulus was selected.

        This method has two modes that return 2D arrays of different
        shapes.

        Returns:
            is_select: A 2D Boolean array indicating the stimuli that
                were selected. By default, this will be a 2D array that
                has the same shape as `stimulus_set`. See the
                `compress` option for non-default behavior.
                shape=(n_trial, n_max_reference + 1) if compress=False
                shape=(n_trial, n_max_select) if compress=True
            compress (optional): A Boolean indicating if the returned
                2D array should be compressed such that the first
                column corresponding to the query is removed, and any
                trailing columns with no selected stimuli are also
                removed. This results in a 2D array with a shape that
                implies the maximum number of selected references.

        """
        is_select = np.zeros(self.stimulus_set.shape, dtype=bool)
        max_n_select = np.max(self.n_select)
        for n_select in range(1, max_n_select + 1):
            locs = np.less_equal(n_select, self.n_select)
            is_select[locs, n_select] = True

        if compress:
            is_select = is_select[:, 1:max_n_select + 1]

        return is_select

    def all_outcomes(self):
        """Inflate stimulus set for all possible outcomes."""
        outcome_idx_list = self.outcome_idx_list
        n_outcome_list = self.config_list['n_outcome'].values
        max_n_outcome = np.max(n_outcome_list)
        n_config = self.config_list.shape[0]

        stimulus_set_expand = -1 * np.ones(
            [self.n_trial, self.max_n_reference + 1, max_n_outcome],
            dtype=np.int32
        )
        for i_config in range(n_config):
            # Identify relevant trials.
            trial_locs = self.config_idx == i_config
            n_trial_config = np.sum(trial_locs)

            config = self.config_list.iloc[i_config]
            n_reference = config['n_reference']
            outcome_idx = outcome_idx_list[i_config]
            n_outcome = outcome_idx.shape[0]
            # Add query index, increment references to accommodate query.
            stimulus_set_idx = np.hstack(
                [np.zeros([n_outcome, 1], dtype=int), outcome_idx + 1]
            )
            curr_stimulus_set_copy = self.stimulus_set[trial_locs, :]
            curr_stimulus_set_expand = -1 * np.ones(
                [n_trial_config, self.max_n_reference + 1, max_n_outcome],
                dtype=int
            )
            for i_outcome in range(n_outcome):
                curr_stimulus_set_idx = stimulus_set_idx[i_outcome, :]
                # Append placeholder indices.
                curr_idx = np.hstack([
                    curr_stimulus_set_idx,
                    np.arange(
                        np.max(curr_stimulus_set_idx) + 1,
                        self.max_n_reference + 1
                    )
                ])
                curr_stimulus_set_expand[:, :, i_outcome] = (
                    curr_stimulus_set_copy[:, curr_idx]
                )
            stimulus_set_expand[trial_locs] = curr_stimulus_set_expand
        return stimulus_set_expand

    @classmethod
    def stack(cls, trials_list):
        """Return a RankTrials object containing all trials.

        The stimulus_set of each SimilarityTrials object is padded first to
        match the maximum number of references of all the objects.

        Arguments:
            trials_list: A tuple of RankTrials objects to be stacked.

        Returns:
            A new RankTrials object.

        """
        # TODO remove any occurence of None from trials_list

        # Determine the maximum number of references.
        max_n_reference = 0
        for i_trials in trials_list:
            if i_trials.max_n_reference > max_n_reference:
                max_n_reference = i_trials.max_n_reference

        # Grab relevant information from first entry in list.
        stimulus_set = pad_2d_array(
            trials_list[0].stimulus_set, max_n_reference + 1
        )
        n_select = trials_list[0].n_select
        is_ranked = trials_list[0].is_ranked
        is_judged = True
        try:
            group_id = trials_list[0].group_id
            agent_id = trials_list[0].agent_id
            session_id = trials_list[0].session_id
            weight = trials_list[0].weight
            rt_ms = trials_list[0].rt_ms
        except AttributeError:
            is_judged = False

        for i_trials in trials_list[1:]:
            stimulus_set = np.vstack((
                stimulus_set,
                pad_2d_array(i_trials.stimulus_set, max_n_reference + 1)
            ))
            n_select = np.hstack((n_select, i_trials.n_select))
            is_ranked = np.hstack((is_ranked, i_trials.is_ranked))
            if is_judged:
                group_id = np.hstack((group_id, i_trials.group_id))
                agent_id = np.hstack((agent_id, i_trials.agent_id))
                session_id = np.hstack((session_id, i_trials.session_id))
                weight = np.hstack((weight, i_trials.weight))
                rt_ms = np.hstack((rt_ms, i_trials.rt_ms))

        if is_judged:
            trials_stacked = RankObservations(
                stimulus_set, n_select=n_select, is_ranked=is_ranked,
                group_id=group_id, agent_id=agent_id, session_id=session_id,
                weight=weight, rt_ms=rt_ms
            )
        else:
            trials_stacked = RankDocket(
                stimulus_set, n_select, is_ranked
            )
        return trials_stacked


class RankDocket(RankTrials):
    """Object that encapsulates unseen trials.

    The attributes and behavior of RankDocket is largely inherited
    from RankTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial. The first column
            is the query stimulus. The remaining, columns indicate
            reference stimuli. Negative integers are used as
            placeholders to indicate non-existent references.
            shape = (n_trial, max(n_reference) + 1)
        n_reference: An integer array indicating the number of
            references in each trial.
            shape = (n_trial,)
        n_select: An integer array indicating the number of references
            selected in each trial.
            shape = (n_trial,)
        is_ranked: A Boolean array indicating which trials require
            reference selections to be ranked.
            shape = (n_trial,)
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_reference',
            'n_select', 'is_ranked'. and 'n_outcome'.
        outcome_idx_list: A list of 2D arrays indicating all possible
            outcomes for a trial configuration. Each element in the
            list corresponds to a trial configuration in config_list.
            Each row of the 2D array indicates one potential outcome.
            The values in the rows are the indices of the the reference
            stimuli (as specified in the attribute `stimulus_set`.

    Notes:
        stimulus_set: The order of the reference stimuli is
            unimportant. As usual, the the first column contains
            indices indicating query stimulus. The remaining columns
            contain indices indicating the reference stimuli in any
            order.
        Unique configurations and configuration IDs are determined by
            'n_reference', 'n_select', and 'is_ranked'.

    Methods:
        save: Save the Docket object to disk.
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set, n_select=None, is_ranked=None):
        """Initialize.

        Arguments:
            stimulus_set: The order of the reference indices is not
                important. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.

        """
        RankTrials.__init__(self, stimulus_set, n_select, is_ranked)

        # Determine unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked
        )

    def subset(self, index):
        """Return subset of trials as a new RankDocket object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RankDocket object.

        """
        return RankDocket(
            self.stimulus_set[index, :], self.n_select[index],
            self.is_ranked[index]
        )

    def _set_configuration_data(self, n_reference, n_select, is_ranked):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Arguments:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = (n_trial,)
            n_select: An integer array indicating the number of
                references selected in each trial.
                shape = (n_trial,)
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = (n_trial,)

        Notes:
            Sets three attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.
            outcome_idx_list: A list of the possible outcomes for each
                trial configuration.

        """
        n_trial = len(n_reference)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {
            'n_reference': n_reference, 'n_select': n_select,
            'is_ranked': is_ranked, 'n_outcome': n_outcome_placeholder
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration ID for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = _possible_rank_outcomes(df_config.iloc[i_config])
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_select == df_config['n_select'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            f = np.array((a, b, c))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config
        self.outcome_idx_list = outcome_idx_list

    def save(self, filepath):
        """Save the RankDocket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="RankDocket")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.close()

    def as_dataset(self, group, all_outcomes=True):
        """Return TensorFlow dataset.

        Arguments:
            group: ND array indicating group membership information for
                each trial.
            all_outcomes (optional): Boolean indicating whether all
                possible outcomes (along third dimension) should be
                included in returned dataset.

        Returns:
            x: A TensorFlow dataset.

        """
        if group is None:
            group = np.zeros([self.n_trial, 1], dtype=np.int32)
        else:
            if group.ndim == 1:
                group = np.expand_dims(group, axis=1)
            group_level_0 = np.zeros([group.shape[0], 1], dtype=np.int32)
            group = np.hstack([group_level_0, group])
        # Return tensorflow dataset.
        if all_outcomes:
            stimulus_set = self.all_outcomes()
            x = {
                'stimulus_set': tf.constant(
                    stimulus_set + 1, dtype=tf.int32
                ),
                'is_select': tf.constant(
                    np.expand_dims(self.is_select(compress=False), axis=2),
                    dtype=tf.bool
                ),
                'group': tf.constant(group, dtype=tf.int32)
            }
        else:
            stimulus_set = np.expand_dims(self.stimulus_set + 1, axis=2)
            x = {
                'stimulus_set': tf.constant(stimulus_set, dtype=tf.int32),
                'is_select': tf.constant(
                    np.expand_dims(self.is_select(compress=False), axis=2),
                    dtype=tf.bool
                ),
                'group': tf.constant(group, dtype=tf.int32)
            }
        return tf.data.Dataset.from_tensor_slices((x))

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Arguments:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        stimulus_set = f["stimulus_set"][()]
        n_select = f["n_select"][()]
        is_ranked = f["is_ranked"][()]
        f.close()
        trials = RankDocket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )
        return trials


class RankObservations(RankTrials):
    """Object that encapsulates seen trials.

    The attributes and behavior of RankObservations are largely inherited
    from RankTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial. The first column
            is the query stimulus. The remaining, columns indicate
            reference stimuli. Negative integers are used as
            placeholders to indicate non-existent references.
            shape = (n_trial, max(n_reference) + 1)
        n_reference: An integer array indicating the number of
            references in each trial.
            shape = (n_trial,)
        n_select: An integer array indicating the number of references
            selected in each trial.
            shape = (n_trial,)
        is_ranked: A Boolean array indicating which trials require
            reference selections to be ranked.
            shape = (n_trial,)
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations.
        outcome_idx_list: A list of 2D arrays indicating all possible
            outcomes for a trial configuration. Each element in the
            list corresponds to a trial configuration in config_list.
            Each row of the 2D array indicates one potential outcome.
            The values in the rows are the indices of the the reference
            stimuli (as specified in the attribute `stimulus_set`.
        group_id: An integer 2D array indicating the group membership
            of each trial. It is assumed that group_id is composed of
            integers from [0, M-1] where M is the total number of
            groups.
            shape = (n_trial, n_col)
        agent_id: An integer array indicating the agent ID of a trial.
            It is assumed that all IDs are non-negative and that
            observations with the same agent ID were judged by a single
            agent.
            shape = (n_trial,)
        session_id: An integer array indicating the session ID of a
            trial. It is assumed that all IDs are non-negative. Trials
            with different session IDs were obtained during different
            sessions.
            shape = (n_trial,)
        weight: An float array indicating the inference weight of each
            trial.
            shape = (n_trial,)
        rt_ms: An array indicating the response time (in milliseconds)
            of the agent for each trial.

    Notes:
        stimulus_set: The order of the reference stimuli is important.
            As usual, the the first column contains indices indicating
            query stimulus. The remaining columns contain indices
            indicating the reference stimuli. An agent's selected
            references are listed first (in order of selection if the
            trial is ranked) and remaining unselected references are
            listed in any order.
        Unique configurations and configuration IDs are determined by
            'group_id' in addition to the usual 'n_reference',
            'n_select', and 'is_ranked' variables.

    Methods:
        subset: Return a subset of judged trials given an index.
        set_group_id: Override the group ID of all trials.
        set_weight: Override the weight of all trials.
        save: Save the observations data structure to disk.

    """

    def __init__(self, stimulus_set, n_select=None, is_ranked=None,
                 group_id=None, agent_id=None, session_id=None, weight=None,
                 rt_ms=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Arguments:
            stimulus_set: The order of reference indices is important.
                An agent's selected references are listed first (in
                order of selection if the trial is ranked) and
                remaining unselected references are listed in any
                order. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
            group_id (optional): An integer 2D array indicating the
                group membership of each trial. It is assumed that
                `group_id` is composed of integers from [0, M-1] where
                M is the total number of groups.
                shape = (n_trial, n_col)
            agent_id: An integer array indicating the agent ID of a
                trial. It is assumed that all IDs are non-negative and
                that observations with the same agent ID were judged by
                a single agent.
                shape = (n_trial,)
            session_id: An integer array indicating the session ID of a
                trial. It is assumed that all IDs are non-negative.
                Trials with different session IDs were obtained during
                different sessions.
                shape = (n_trial,)
            weight (optional): A float array indicating the inference
                weight of each trial.
                shape = (n_trial,1)
            rt_ms(optional): An array indicating the response time (in
                milliseconds) of the agent for each trial.
                shape = (n_trial,1)

        """
        RankTrials.__init__(self, stimulus_set, n_select, is_ranked)

        # Handle default settings.
        if group_id is None:
            group_id = np.zeros((self.n_trial), dtype=np.int32)  # TODO
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = group_id

        if agent_id is None:
            agent_id = np.zeros((self.n_trial), dtype=np.int32)
        else:
            agent_id = self._check_agent_id(agent_id)
        self.agent_id = agent_id

        if session_id is None:
            session_id = np.zeros((self.n_trial), dtype=np.int32)
        else:
            session_id = self._check_session_id(session_id)
        self.session_id = session_id

        if weight is None:
            weight = np.ones((self.n_trial))
        else:
            weight = self._check_weight(weight)
        self.weight = weight

        if rt_ms is None:
            rt_ms = -np.ones((self.n_trial))
        else:
            rt_ms = self._check_rt(rt_ms)
        self.rt_ms = rt_ms

        # Determine unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked, group_id)

    def _check_group_id(self, group_id):
        """Check the argument group_id."""
        group_id = group_id.astype(np.int32)
        # Check shape agreement.
        if not (group_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'group_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = group_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'group_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return group_id

    def _check_agent_id(self, agent_id):
        """Check the argument agent_id."""
        agent_id = agent_id.astype(np.int32)
        # Check shape agreement.
        if not (agent_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'agent_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = agent_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'agent_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return agent_id

    def _check_session_id(self, session_id):
        """Check the argument session_id."""
        session_id = session_id.astype(np.int32)
        # Check shape agreement.
        if not (session_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'session_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = session_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'session_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return session_id

    def _check_weight(self, weight):
        """Check the argument weight."""
        weight = weight.astype(np.float)
        # Check shape agreement.
        if not (weight.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'weight' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return weight

    def _check_rt(self, rt_ms):
        """Check the argument rt_ms."""
        rt_ms = rt_ms.astype(np.float)
        # Check shape agreement.
        if not (rt_ms.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'rt_ms' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return rt_ms

    def subset(self, index):
        """Return subset of trials as a new RankObservations object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RankObservations object.

        """
        return RankObservations(
            self.stimulus_set[index, :], n_select=self.n_select[index],
            is_ranked=self.is_ranked[index], group_id=self.group_id[index],
            agent_id=self.agent_id[index], session_id=self.session_id[index],
            weight=self.weight[index], rt_ms=self.rt_ms[index]
        )

    def _set_configuration_data(
                self, n_reference, n_select, is_ranked, group_id,
                session_id=None):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Arguments:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = (n_trial,)
            n_select: An integer array indicating the number of
                references selected in each trial.
                shape = (n_trial,)
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = (n_trial,)
            group_id: An integer array indicating the group membership
                of each trial. It is assumed that group is composed of
                integers from [0, M-1] where M is the total number of
                groups. Separate attention weights are inferred for
                each group.
                shape = (n_trial,)
            session_id: An integer array indicating the session ID of
                a trial. It is assumed that observations with the same
                session ID were judged by a single agent. A single
                agent may have completed multiple sessions.
                shape = (n_trial,)

        Notes:
            Sets three attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.
            outcome_idx_list: A list of the possible outcomes for each
                trial configuration.

        """
        n_trial = len(n_reference)

        if session_id is None:
            session_id = np.zeros((n_trial), dtype=np.int32)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {
            'n_reference': n_reference, 'n_select': n_select,
            'is_ranked': is_ranked, 'group_id': group_id,
            'session_id': session_id, 'n_outcome': n_outcome_placeholder
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration index for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = _possible_rank_outcomes(df_config.iloc[i_config])
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_select == df_config['n_select'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            d = (group_id == df_config['group_id'].iloc[i_config])
            e = (session_id == df_config['session_id'].iloc[i_config])
            f = np.array((a, b, c, d, e))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config
        self.outcome_idx_list = outcome_idx_list

    def set_group_id(self, group_id):
        """Override the existing group_ids.

        Arguments:
            group_id: The new group IDs. Can be an integer or an array
                of integers with shape=(self.n_trial,).

        """
        if np.isscalar(group_id):
            group_id = group_id * np.ones((self.n_trial), dtype=np.int32)
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = copy.copy(group_id)

        # Re-derive unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked, group_id)

    def set_weight(self, weight):
        """Override the existing weights.

        Arguments:
            weight: The new weight. Can be an float or an array
                of floats with shape=(self.n_trial,).

        """
        if np.isscalar(weight):
            weight = weight * np.ones((self.n_trial), dtype=np.int32)
        else:
            weight = self._check_weight(weight)
        self.weight = copy.copy(weight)

    def save(self, filepath):
        """Save the RankObservations object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="RankObservations")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.create_dataset("group_id", data=self.group_id)
        f.create_dataset("agent_id", data=self.agent_id)
        f.create_dataset("session_id", data=self.session_id)
        f.create_dataset("weight", data=self.weight)
        f.create_dataset("rt_ms", data=self.rt_ms)
        f.close()

    def as_dataset(self, all_outcomes=True):
        """Format necessary data as Tensorflow.data.Dataset object.

        Returns:
            ds_obs: The data necessary for inference, formatted as a
            tf.data.Dataset object.

        """
        # NOTE: Should not use single dimension inputs. Add a singleton
        # dimensions if necessary, because restoring a SavedModel adds
        # singleton dimensions on the call signautre for inputs that only
        # have one dimension. Add the singleton dimensions here solves the
        # problem.
        # NOTE: We use stimulus_set + 1, since TensorFlow requires "0", not
        # "-1" to indicate a masked value.
        # NOTE: The dimensions of inputs are expanded to have an additional
        # singleton third dimension to indicate that there is only one outcome
        # that we are interested for each trial.
        group_level_0 = np.zeros([self.group_id.shape[0]], dtype=np.int32)
        if all_outcomes:
            stimulus_set = self.all_outcomes()
            x = {
                'stimulus_set': stimulus_set + 1,
                'is_select': np.expand_dims(
                    self.is_select(compress=False), axis=2
                ),
                'group': np.stack(
                    (group_level_0, self.group_id, self.agent_id), axis=-1
                )
            }
            # NOTE: The outputs `y` indicate a one-hot encoding of the outcome
            # that occurred.
            y = np.zeros([self.n_trial, stimulus_set.shape[2]])
            y[:, 0] = 1
        else:
            x = {
                'stimulus_set': np.expand_dims(self.stimulus_set + 1, axis=2),
                'is_select': np.expand_dims(
                    self.is_select(compress=False), axis=2
                ),
                'group': np.stack(
                    (group_level_0, self.group_id, self.agent_id), axis=-1
                )
            }
            # NOTE: The outputs `y` indicate a sparse encoding of the outcome
            # that occurred.
            y = np.zeros([self.n_trial])

        y = tf.constant(y, dtype=K.floatx())

        # Observation weight.
        w = tf.constant(self.weight, dtype=K.floatx())

        # Create dataset.
        ds_obs = tf.data.Dataset.from_tensor_slices((x, y, w))
        return ds_obs

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Arguments:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        stimulus_set = f["stimulus_set"][()]
        n_select = f["n_select"][()]
        is_ranked = f["is_ranked"][()]
        group_id = f["group_id"][()]

        # For backwards compatability.
        if "weight" in f:
            weight = f["weight"][()]
        else:
            weight = np.ones((len(n_select)))
        if "rt_ms" in f:
            rt_ms = f["rt_ms"][()]
        else:
            rt_ms = -np.ones((len(n_select)))
        if "agent_id" in f:
            agent_id = f["agent_id"][()]
        else:
            agent_id = np.zeros((len(n_select)))
        if "session_id" in f:
            session_id = f["session_id"][()]
        else:
            session_id = np.zeros((len(n_select)))
        f.close()

        trials = RankObservations(
            stimulus_set, n_select=n_select, is_ranked=is_ranked,
            group_id=group_id, agent_id=agent_id, session_id=session_id,
            weight=weight, rt_ms=rt_ms
        )
        return trials


def _possible_rank_outcomes(trial_configuration):
    """Return the possible outcomes of a ranked trial configuration.

    Arguments:
        trial_configuration: A trial configuration Pandas Series.

    Returns:
        An 2D array indicating all possible outcomes where the values
            indicate indices of the reference stimuli. Each row
            corresponds to one outcome. Note the indices refer to
            references only and does not include an index for the
            query. Also note that the unpermuted index is returned
            first.

    """
    n_reference = int(trial_configuration['n_reference'])
    n_select = int(trial_configuration['n_select'])

    reference_list = range(n_reference)

    # Get all permutations of length n_select.
    perm = permutations(reference_list, n_select)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, n_reference), dtype=np.int32)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_select] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(n_reference)
        for i_selected in range(n_select):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_select:] = dummy_idx

    return outcomes
