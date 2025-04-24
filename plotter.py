"""
Plotter Scripts 

This script is used to plot the results of the models trained on the data.
This script will need to be modified to work with new results and data formats

Author: Paul Gariy
Date: 2025-05-22
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import pickle
import re

# Project functions
from main import *
from parser import *
from pfam_loader import *


def plot_confusion_matrix(all_ytrue, all_ypred, network, classes=46):
    """
    Function to plot confusion matrix

    Args:
    all_ytrue: true labels
    all_ypred: predicted labels
    """
    # Compute confusion matrix
    cm = confusion_matrix(all_ytrue, all_ypred, labels=range(classes))

    title = f"Confusion Matrix for {network}"
    cmap = plt.get_cmap("Blues")

    class_labels = [i for i in range(classes)]
    plt.figure(figsize=(12, 6))
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(classes), class_labels, rotation=45)
    plt.yticks(np.arange(classes), class_labels)
    plt.colorbar()

    # Add value annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"./results/confusion_matrix_{network}.png")
    plt.close()


def plot_accuracy_bars(rnn_results, gru_results, attn_results):
    """
    Function to plot accuracy bars for each network grouped by rotation

    Args:
    rnn_results: list of RNN results
    gru_results: list of GRU results
    attn_results: list of ATTN results
    """
    # get validation accuracy for each fold
    rotation_acc = {
        "r0": [],
        "r1": [],
        "r2": [],
        "r3": [],
        "r4": [],
    }
    # get the results for each fold for each network with RNN index 0, GRU index 1, ATTN index 2
    for i in range(0, 5):
        # load the results for each fold for each network
        with open(rnn_results[i], "rb") as f:
            results = pickle.load(f)
            rotation_acc["r%d" % (i)].append(results["final_test_accuracy"])
        with open(gru_results[i], "rb") as f:
            results = pickle.load(f)
            rotation_acc["r%d" % (i)].append(results["final_test_accuracy"])
        with open(attn_results[i], "rb") as f:
            results = pickle.load(f)
            rotation_acc["r%d" % (i)].append(results["final_test_accuracy"])

    # convert to numpy array
    rotation_acc = np.array(
        [
            rotation_acc["r0"],
            rotation_acc["r1"],
            rotation_acc["r2"],
            rotation_acc["r3"],
            rotation_acc["r4"],
        ]
    )
    print(rotation_acc)
    # plot grouped bar chart grouped by rotation with network labels
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    x = np.arange(len(rotation_acc))
    ax.bar(x - bar_width, rotation_acc[:, 0], width=bar_width, label="RNN")
    ax.bar(x, rotation_acc[:, 1], width=bar_width, label="GRU")
    ax.bar(x + bar_width, rotation_acc[:, 2], width=bar_width, label="ATTN")

    ax.set_xlabel("Rotation")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Figure 3: Validation Accuracy for each Network")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rotation {i}" for i in range(len(rotation_acc))])
    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig("./results/figures/accuracy_bars.png")
    plt.close()


def plot_val_accuracy_epochs(results_gru, results_attn):
    """
    Function to plot validation accuracy vs epochs for GRU and ATTN networks

    Args:
    results_gru: list of GRU results
    results_attn: list of ATTN results
    """
    # create plot
    plt.figure(figsize=(10, 6))
    plt.title(
        "Validation Set Sparse Categorical Accuracy GRU vs Validation Set Sparse Categorical Accuracy ATTN"
    )
    plt.xlabel("Val Categorical Accuracy GRU")
    plt.ylabel("Val Categorical Accuracy ATTN")
    # Get training history for GRU and ATTN and plot validation accuracy per epoch for each fold
    for i in range(0, 5):
        with open(results_gru[i], "rb") as f:
            results = pickle.load(f)
            history = results["history"]
            val_acc_gru = history["val_sparse_categorical_accuracy"]
            epochs_gru = range(1, len(val_acc_gru) + 1)
        with open(results_attn[i], "rb") as f:
            results = pickle.load(f)
            history = results["history"]
            val_acc_att = history["val_sparse_categorical_accuracy"]
            epochs_att = range(1, len(val_acc_att) + 1)

        # pad the val acc with least epochs
        if len(epochs_gru) < len(epochs_att):
            val_acc_gru = np.pad(
                val_acc_gru, (0, len(epochs_att) - len(epochs_gru)), "edge"
            )
            epochs_gru = range(1, len(val_acc_att) + 1)
        elif len(epochs_att) < len(epochs_gru):
            val_acc_att = np.pad(
                val_acc_att, (0, len(epochs_gru) - len(epochs_att)), "edge"
            )
            epochs_att = range(1, len(val_acc_gru) + 1)
        # Plot the val accuracy GRU vs val accuracy ATTN

        plt.plot(val_acc_gru, val_acc_att, label=f"Rotation {i}")
    # Add legend and grid
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # Save the figure
    plt.savefig("./results/figures/fig2.png")


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # To change args for each network
    networks = ["RNN", "GRU", "ATTN"]
    # best performing network from test set accuracy
    best_network = "GRU"
    # Paths
    results_paths = []
    model_paths = []

    # Loop through each network and get the results
    for n in networks:
        args.network = n
        # get fbase name from args for each fold
        for i in range(0, 5):
            args.rotation = i
            fbase = generate_filename(args)
            fname = "%s_results.pkl" % (fbase)
            modelname = "%s_model.h5" % (fbase)
            # check if file exists then append to results_paths
            if os.path.exists(fname) and os.path.exists(modelname):
                results_paths.append(fname)
                model_paths.append(modelname)
            else:
                print("File does not exist: ", fname)

    # select the best performing networks result paths
    # best_results_paths = [path for path in results_paths if best_network in path]
    best_results_model = [path for path in model_paths if best_network in path]

    # overall preds and true labels
    all_ytrue = []
    all_ypred = []
    # for each  rotation load the model and get the predictions and append to the true and predicted labels
    for i in range(0, 5):
        # load the model
        model = tf.keras.models.load_model(best_results_model[i])

        # load data
        data_dict = load_rotation(basedir="./data", rotation=i, version="B")
        if data_dict is None:
            print(f"Data path {args.dataset} does not exist. Exiting.")

        # Convert to 3TF DataSets
        dataset_train, dataset_valid, dataset_test = create_tf_datasets(
            dat=data_dict,
            batch=64,
            prefetch=-1,
            shuffle=100,
            repeat=True,
            cache=" ",
        )

        # get number of classes
        n_classes = data_dict["n_classes"]

        # store true labels and predicted labels
        ytrue = []
        ypred = []

        # predict on the test set
        for x, y in dataset_test:
            # Get the predictions
            y_pred = model.predict(x)
            # Convert to numpy arrays
            ytrue.append(tf_to_numpy(y))
            ypred.append(tf_to_numpy(y_pred))

        # Concatenate the true and predicted labelz
        all_ytrue.append(np.concatenate(ytrue, axis=0))
        all_ypred.append(np.concatenate(ypred, axis=0))

    # Convert to numpy arrays
    all_ytrue = np.array(all_ytrue)
    all_ypred = np.array(all_ypred)

    # Convert to 1D arrays
    all_ytrue = all_ytrue.flatten()
    all_ypred = all_ypred.flatten()

    # find num classes
    n_classes = len(np.unique(all_ytrue))

    # plot confusion matrix
    plot_confusion_matrix(all_ytrue, all_ypred, args)

    # get results for each network
    rnn_results = [path for path in results_paths if "RNN" in path]
    gru_results = [path for path in results_paths if "GRU" in path]
    attn_results = [path for path in results_paths if "ATTN" in path]

    plot_accuracy_bars(rnn_results, gru_results, attn_results)
    plot_val_accuracy_epochs(gru_results, attn_results)
