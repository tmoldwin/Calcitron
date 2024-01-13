import random
import numpy as np
from numpy.random import choice

def noisify(pattern, flip_prob=0.1):
    return np.array([bit if (np.random.rand() > flip_prob) else 1 - bit for bit in pattern])

def signal_noise_gen_old(patterns, pattern_probs, P, flip_prob):
    patterns = np.atleast_2d(patterns)
    (patterns)
    patterns = np.vstack([np.zeros_like(patterns[0]), patterns])  # add the empty pattern
    pattern_probs = pattern_probs.append(1 - np.sum(pattern_probs))
    draw = choice(range(len(patterns)), P,
                  p=pattern_probs)
    noised_draw = np.array([noisify(patterns[i], flip_prob) if i == 0 else patterns[i] for i in draw])
    return noised_draw

def pattern_gen(N, P, density):
    (P)
    pos_N = int(density * N)
    neg_N = N - pos_N
    return np.squeeze([np.random.permutation([0] * neg_N + [1] * pos_N) for i in range(P)])

def signal_noise_gen(N, P, signal_prob, density):
    signal_pattern = pattern_gen(N, 1, density)
    (signal_pattern)
    pattern_probs = [1 - signal_prob, signal_prob]
    draw = choice(range(2), P, p=pattern_probs)
    input_patterns = np.squeeze([pattern_gen(N, 1, density) if i == 0 else signal_pattern for i in draw])
    (input_patterns.shape)
    return signal_pattern, draw, input_patterns

def random_inputs_all(N, P):
    pattern3 = np.random.permutation([0] * int(0.75 * N) + [1] * int(0.25 * N))
    pattern2 = np.random.permutation([0] * int(0.92 * N) + [1] * int(0.08 * N))
    pattern1 = np.random.permutation([0] * int(0.95 * N) + [1] * int(0.05 * N))
    patterns = [pattern1, pattern2, pattern3]
    #               0         1        2
    probs = [0.4, 0.3, 0.3]
    pat_nums = np.random.permutation(sum([[i] * int(probs[i] * P) for i in range(len(patterns))], []))
    local_inputs = [patterns[pat_nums[i]] for i in range(P)]
    return pat_nums, local_inputs

def pattern_mixer(patterns, P, pattern_probs = None):
    '''
    Given a vector of input patterns, randomly draws P
    from them according with probability pattern_probs.
    Returns the pattern indices and patterns
    '''
    pattern_nums = choice(range(len(patterns)), P, p=pattern_probs)
    local_inputs = np.array([patterns[i] for i in pattern_nums])
    return pattern_nums, np.squeeze(local_inputs)


def generate_perceptron_patterns(P, N, sparsity):
    # Initialize the random number generator
    # Generate the patterns
    patterns = []
    for i in range(P):
        random.seed(1)
        rng = random.Random()
        # Generate a random binary pattern with the specified sparsity
        pattern = [1 if rng.random() < sparsity else 0 for _ in range(N)]
        # Store the pattern and its label
        patterns.append(pattern)
    labels = [1] * int(P/2) + [0] * int(P/2)
    random.seed(1)
    random.shuffle(labels)
    return patterns, labels