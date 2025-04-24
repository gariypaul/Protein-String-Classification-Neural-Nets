"""
Argument parser for PFAM String Classification Experiment

Author: Paul Gariy
Date: 2025-05-22
Description: This script defines a command-line argument parser for a PFAM string classification experiment. It includes options for configuring the experiment, such as network architecture, training parameters, and regularization settings. The parser is designed to be used with the argparse library in Python.
"""

import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='STRINGCLASS', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='String Classification', help='WandB project name')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')

    # High-level experiment configuration
    parser.add_argument('--network', type=str, default='RNN', help='Network name')
    parser.add_argument('--dataset', type=str, default='/scratch/fagg/pfam', help='Data set directory')
    parser.add_argument('--ntrain_folds', type=int, default=6, help='Maximum number of training folds')
    parser.add_argument('--nvalid_folds', type=int, default=1, help='Maximum number of validation folds')
    parser.add_argument('--ntest_folds', type=int, default=1, help='Maximum number of testing folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='Number of steps per epoch')
    
    
    # Specific experiment configuration
    parser.add_argument('--rotation', type=int, default=0, help='Data fold to use within dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=5e-5, help="Learning rate")


    #Architecture parameters
    parser.add_argument('--nhidden_layers', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--ndense_layers', nargs='+', type=int, default=[100, 5], help='Number of dense units per layer (sequence of ints)')
    parser.add_argument('--dense_activations', nargs='+', type=str, default=['elu', 'elu'], help='Activation function for dense layers (sequence of strings)')
    parser.add_argument('--activation', type=str, default='elu', help='Activation function for hidden dense layers')
    parser.add_argument('--batch_normalization', action='store_true', default=None, help='Use batch normalization')
    parser.add_argument('--embedding_dimension', type=int, default=128, help='Embedding dimension for categorical features')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate for hidden layers')
    parser.add_argument('--dropout_input', type=float, default=None, help='Dropout rate for input layer')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")
    parser.add_argument('--recurrent_L1_regularization', '--rl1', type=float, default=None, help="Recurrent L1 regularization parameter")
    parser.add_argument('--recurrent_L2_regularization', '--rl2', type=float, default=None, help="Recurrent L2 regularization parameter")


    # Early stopping
    parser.add_argument('--earlystop', action='store_true', default=True, help='Use early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=-1, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=" ", help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    #Post training parameters
    parser.add_argument('--render', action='store_true', default=False , help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')


    return parser