
"""
This script performs a one-way ANOVA test on the accuracy of three different neural network architectures: RNN, GRU, and MHA.

The data is hardcoded and will need editing if the data changes.

The script uses the `scipy.stats` library to perform the ANOVA test and prints the F-value and p-value.

Author: Paul Gariy
Date: 2025-05-22
Description: This script performs a one-way ANOVA test on the accuracy of three different neural network architectures: RNN, GRU, and MHA.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd



# Data (from your table)
rnn_accuracy = np.array([0.98471475, 0.99293643, 0.99275655, 0.74119174, 0.96579874])
gru_accuracy = np.array([0.9942776, 0.99623954, 0.99453568, 0.99379736, 0.99710113])
mha_accuracy = np.array([0.99768561, 0.99847549, 0.99867839, 0.99885607, 0.99883032])

# Combine data for ANOVA
accuracy = np.concatenate([rnn_accuracy, gru_accuracy, mha_accuracy])
networks = np.repeat(["RNN", "GRU", "MHA"], 5)

# Perform ANOVA
fvalue, pvalue = stats.f_oneway(rnn_accuracy, gru_accuracy, mha_accuracy)
print(f"ANOVA F-value: {fvalue}, p-value: {pvalue}")

