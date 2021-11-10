"""Hacky script to export model."""

import argparse
import os

# don't use GPU if busy
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import h5py
import numpy as np
import psiz
import tensorflow as tf
import tensorflow.keras as tfk


BATCH_SIZE = 100


def _load_catalog(file_name):
    h5_file = h5py.File(file_name, "r")
    stimulus_id = h5_file["stimulus_id"][()]
    # pylint: disable=no-member
    stimulus_filepath = h5_file["stimulus_filepath"][:]
    class_id = h5_file["class_id"][()]

    try:
        class_map_class_id = h5_file["class_map_class_id"][()]
        class_map_label = h5_file["class_map_label"][()]
        class_label_dict = {}
        for idx in np.arange(len(class_map_class_id)):
            class_label_dict[class_map_class_id[idx]] = class_map_label[idx].decode("ascii")
    except KeyError:
        class_label_dict = None

    h5_file.close()

    return stimulus_id, stimulus_filepath, class_id, class_label_dict


def get_index_mapping(directory_name):
    stimulus_id, stimulus_filepath, _, _ = _load_catalog(
        os.path.join(directory_name, "val", "catalogs", "psiz0.4.1", "catalog.hdf5")
    )

    index_mapping = {}
    for old_index, file_name in zip(stimulus_id, stimulus_filepath):
        file_name = file_name.decode("utf-8")
        i, j = file_name.rfind("_"), file_name.rfind(".")
        new_index = int(file_name[(i + 1) : j]) - 1
        index_mapping[new_index] = old_index

    index_mapping = [index_mapping[i] for i in range(len(index_mapping))]

    return index_mapping


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-directory-name", help="Path to raw data.", required=True)
    parser.add_argument("--model-directory-name", help="Path to model.", required=True)
    args = parser.parse_args()

    # Load model and extract embeddings.
    model = tf.keras.models.load_model(args.model_directory_name)
    embeddings = model.stimuli.posterior.loc.numpy()[1:]

    # Remap the embeddings so their indexes correspond to file names (0 -> ILSVRC2012_val_00000001.JPEG, 1 -> ILSVRC2012_val_00000002.JPEG etc).
    index_mapping = get_index_mapping(args.raw_data_directory_name)
    embeddings = embeddings[index_mapping]

    # Make new TF model mapping indexes to the corresponding embeddings.
    index_input = tfk.layers.Input(shape=(), batch_size=BATCH_SIZE, dtype=tf.int32)
    x = tfk.layers.Embedding(
        embeddings.shape[0], embeddings.shape[1], embeddings_initializer=tfk.initializers.Constant(embeddings)
    )(index_input)
    converted_model = tfk.models.Model(index_input, x, name="converted_model")

    # Save new TF model in same directory as original.
    output_model_directory_name = args.model_directory_name + "-converted"
    converted_model.save(output_model_directory_name)
