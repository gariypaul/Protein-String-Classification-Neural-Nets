"""
Main script for running the experiment
This script is responsible for loading the data, building the model, training the model, and saving the results.
It uses the TensorFlow and Keras libraries for building and training the model.
It also uses the argparse library for parsing command line arguments and the wandb library for logging the results.
It uses the pfam_loader, parser, neuralnets, args_check modules for loading the data, parsing the arguments, building the model, checking the arguments, and controlling the job respectively.

Author: Paul Gariy
Date: 2025-05-22

"""




# external libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import argparse
import pickle
import wandb
import socket

# project code
from pfam_loader import *
from parser import *
from neuralnets import *
from args_check import *



def generate_filename(args):
    """
    Generate a filename based on the arguments provided
    """
    # hidden_layers
    hidden_str = "_".join(str(x) for x in args.nhidden_layers)
    # dropout
    if args.dropout is None:
        dropout_str = ""
    else:
        dropout_str = "drop_%0.3f_" % (args.dropout)
    # L1 regularization
    if args.L1_regularization is None:
        regularizer_l1_str = ""
    else:
        regularizer_l1_str = "L1_%0.6f_" % (args.L1_regularization)

    network_type = args.network
    # L2 regularization
    if args.L2_regularization is None:
        regularizer_l2_str = ""
    else:
        regularizer_l2_str = "L2_%0.6f_" % (args.L2_regularization)
    return "%s/%s_Fold_%d_Hidden%s_%s_%s" % (
        args.results_path,
        network_type,
        args.rotation,
        hidden_str,
        dropout_str,
        regularizer_l1_str + regularizer_l2_str,
    )


def execute_exp(args: argparse.ArgumentParser = None):
    """
    Execute the experiment with the given arguments

    """
    # Check if args are provided else return and exit experiment
    if args is None:
        print("No arguments provided. Exiting.")
        return

    # initialize the experiment
    print("Initializing experiment")
    # get output base name
    fname_base = generate_filename(args)

    # Check if the results path exists, if not create it
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        print(f"Results path {args.results_path} created.")
    else:
        print(f"Results path {args.results_path} already exists.")

    # Check if the data path exists, if not return and exit experiment
    if not os.path.exists(args.dataset):
        print(f"Data path {args.dataset} does not exist. Exiting.")
        return

    ################################################################################
    # DATASET LOADING                                                              #
    ################################################################################
    data_frames = load_pfam_dataset(
        basedir=args.dataset,
        rotation=args.rotation,
        nfolds=5,  # Assuming 5 folds as per the loader's default
        ntrain_folds=3,  # Assuming 3 training folds as per the loader's default
        version="B"  # Assuming version "B" as in your original load_rotation call
    )
    if data_frames is None:
        print(f"Error loading dataset from {args.dataset}. Exiting.")
        return

    # Prepare the data dictionary in the format expected by create_tf_datasets
    data_dict = {
        'ins_train': data_frames['train']['string'].values,
        'outs_train': data_frames['train']['label'].values,
        'ins_valid': data_frames['valid']['string'].values,
        'outs_valid': data_frames['valid']['label'].values,
        'ins_test': data_frames['test']['string'].values,
        'outs_test': data_frames['test']['label'].values,
    }

    dataset_train, dataset_valid, dataset_test = create_tf_datasets(
        dat=data_dict,
        batch=args.batch_size,
        prefetch=-1,
        shuffle=args.shuffle,
        repeat=(args.steps_per_epoch is not None),
        cache=args.cache,
    )

    # Additional information for the model
    len_max = data_dict["len_max"]
    n_tokens = data_dict["n_tokens"]
    n_classes = data_dict["n_classes"]

    #################################################################################
    # Build the model                                                               #
    #################################################################################

    # Create the model based on the network type
    model = build_network(
    len_max = len_max,
    n_tokens = n_tokens,
    nhidden_layers = args.nhidden_layers,
    ndense_layers = args.ndense_layers,
    dense_activations = args.dense_activations,
    dropout = args.dropout,
    L1 = args.L1_regularization,
    L2 = args.L2_regularization,
    recurrent_L1 = args.recurrent_L1_regularization,
    recurrent_L2 = args.recurrent_L2_regularization,
    activation = args.activation,
    num_classes = n_classes,
    embedding_dimension = args.embedding_dimension,
    network_type = args.network,
    num_heads = args.num_heads,
    batch_normalization = args.batch_normalization,
)

    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
    # Compile the model
    model.compile(
        optimizer=opt,
        loss=SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Print the model summary
    model.summary()

    if args.verbose > 1:
        print(args)

    # results output filename
    fname_out = "%s_results.pkl" % (fname_base)

    # render model plots if requested from args
    if args.render:
        render_modelname = "%s_modelplot.png" % (fname_base)
        plot_model(
            model, to_file=render_modelname, show_shapes=True, show_layer_names=True
        )
        print(f"Model plot saved to {render_modelname}")

        # Check if the model plot was created successfully
        if os.path.exists(render_modelname):
            print(f"Model plot {render_modelname} created successfully.")
        else:
            print(f"Model plot {render_modelname} not created. Exiting.")
            return

    # check if experiment already exists
    if os.path.exists(fname_out) and not args.force:
        print(f"Experiment {fname_out} already exists. Exiting.")
        return

    # init wandb
    run = wandb.init(
        project=args.project,
        name=fname_base,
        config=args,
        notes="Experiment for pfam.%s" % (fname_base)
    )

    # log hostname
    wandb.log({"hostname": socket.gethostname()})

    if args.render:
        wandb.log({"model_plot": wandb.Image(render_modelname)})

    # callbacks
    cbs = []

    # Early stopping callback
    if args.earlystop:
        e_stop = EarlyStopping(
            patience=args.patience,
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
            min_delta=0.001,
        )

    # wandb callback
    wandb_cb = wandb.keras.WandbMetricsLogger()
    # add wandb callbacks to list
    cbs.append(wandb_cb)

    # add callbacks to list
    
    if args.earlystop:
        cbs.append(e_stop)

    if args.verbose >= 3:
        print("Fitting model...")

    # Fit the model
    history = model.fit(
        dataset_train,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=dataset_valid,
        callbacks=cbs,
    )

    results = {}
    # save exp data
    results["args"] = args
    results["len_max"] = len_max
    results["n_tokens"] = n_tokens
    results["n_classes"] = n_classes
    results["fname_base"] = fname_base
    ############### Validation Set ####################
    print("#################")
    print("VALIDATION SET")
    results["predict_validation"] = model.predict(dataset_valid)
    results["predict_validation_eval"] = model.evaluate(dataset_valid)
    results["final_validation_loss"] = results["predict_validation_eval"][0]
    results["final_validaction_acuracy"] = results["predict_validation_eval"][1]
    wandb.log({"final_validation_loss": results["predict_validation_eval"][0]})
    wandb.log({"final_validation_accuracy": results["predict_validation_eval"][1]})
    # results['predict_validation'] = np.argmax(results['predict_validation'], axis=1)

    ################# Test Set ####################
    print("#################")
    print("TEST SET")
    results["predict_test"] = model.predict(dataset_test)
    results["predict_test_eval"] = model.evaluate(dataset_test)
    results["final_test_loss"] = results["predict_test_eval"][0]
    results["final_test_accuracy"] = results["predict_test_eval"][1]
    wandb.log({"final_test_loss": results["predict_test_eval"][0]})
    wandb.log({"final_test_accuracy": results["predict_test_eval"][1]})
    # results['predict_test'] = np.argmax(results['predict_test'], axis=1)

    results["history"] = history.history
    # save results in pickle file
    with open(fname_out, "wb") as fp:
        pickle.dump(results, fp)
        print(f"Results saved to {fname_out}")

    # save model
    if args.save_model:
        model.save("%s_model.h5" % (fname_base))
        print(f"Model saved to {fname_base}_model.h5")

    # finish wandb run
    wandb.finish()
    print("Experiment finished")


if __name__ == "__main__":
    # parse arguments
    parser = create_parser()
    args = parser.parse_args()
    # check arg for validity
    check_args(args)

    if args.verbose > 2:
        print("Arguments parsed successfully")

    # GPU check
    visible_devices = tf.config.get_visible_devices("GPU")
    n_visible_devices = len(visible_devices)
    print("GPUS:", visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("We have %d GPUs\n" % n_visible_devices)
    else:
        print("NO GPU")

    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task // 2)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task // 2)

    # execute the experiment
    execute_exp(args)
