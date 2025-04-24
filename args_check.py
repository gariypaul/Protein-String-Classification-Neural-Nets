import os
"""
Check arguments for the experiment.
This script validates the command-line arguments provided for the experiment.
It checks for the existence of necessary files, the validity of numerical values, and the consistency of parameters.
It raises exceptions with informative messages if any checks fail.

Author: Paul Gariy
Date: 2025-05-22
"""

def check_args(args):
    """
    Validate necessary arguments for the experiment.
    Raises an exception on failure.
    """
    # Dataset path
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    # Results directory
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        print(f"Created results directory: {args.results_path}")
    else:
        print(f"Results directory already exists: {args.results_path}")

    # Folds and rotation
    for attr in ("ntrain_folds", "nvalid_folds", "ntest_folds", "rotation"):
        if getattr(args, attr) is None:
            raise ValueError(f"Please provide --{attr} argument.")

    # Learning rate and epochs
    if args.lrate is None or args.lrate <= 0:
        raise ValueError("Please provide a positive --lrate (learning rate).")
    if args.epochs is None or args.epochs <= 0:
        raise ValueError("Please provide a positive --epochs value.")

    # Network type
    if args.network not in ("RNN", "GRU", "ATTN"):
        raise ValueError("--network must be one of: RNN, GRU, ATTN.")

    # Hidden/dense layer specs
    if not args.nhidden_layers:
        raise ValueError(
            "--nhidden_layers must specify at least one hidden layer size."
        )
    if not args.ndense_layers:
        raise ValueError("--ndense_layers must specify at least one dense layer size.")
    if len(args.ndense_layers) != len(args.dense_activations):
        raise ValueError(
            "The number of --dense_activations must match the number of --ndense_layers."
        )

    # Embedding and attention hyperparameters
    if args.embedding_dimension is None or args.embedding_dimension <= 0:
        raise ValueError("--embedding_dimension must be a positive integer.")
    if args.network == "ATTN" and (args.num_heads is None or args.num_heads <= 0):
        raise ValueError("--num_heads must be a positive integer for ATTN networks.")

    # Dropout ranges
    if args.dropout is not None and not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be between 0.0 and 1.0.")
    if args.dropout_input is not None and not (0.0 <= args.dropout_input < 1.0):
        raise ValueError("--dropout_input must be between 0.0 and 1.0.")

    # Regularization non-negativity
    for reg_name in (
        "L1_regularization",
        "L2_regularization",
        "recurrent_L1_regularization",
        "recurrent_L2_regularization",
    ):
        val = getattr(args, reg_name)
        if val is not None and val < 0:
            raise ValueError(f"--{reg_name} must be non-negative.")

    # Early stopping
    if args.earlystop:
        if args.patience is None or args.patience < 0:
            raise ValueError(
                "--patience must be a non-negative integer for early stopping."
            )
        if args.min_delta is None or args.min_delta < 0:
            raise ValueError("--min_delta must be non-negative for early stopping.")
        if not args.monitor:
            raise ValueError("--monitor must be specified for early stopping.")

    # Batch and data pipeline
    if args.batch_size is None or args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if args.shuffle is None or args.shuffle < 0:
        raise ValueError("--shuffle must be non-negative (shuffle buffer size).")
    if args.prefetch is None or args.prefetch < 0:
        raise ValueError(
            "--prefetch must be non-negative (number of batches to prefetch)."
        )

    return True


if __name__ == "__main__":
    from parser import create_parser

    parser = create_parser()
    args = parser.parse_args()
    try:
        if check_args(args):
            print("✅ Arguments are valid!")
    except Exception as e:
        print(f"❌ Argument check failed: {e}")
        exit(2)
