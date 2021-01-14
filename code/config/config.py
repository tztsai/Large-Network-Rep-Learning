import torch

# graph embedding
D = 8                   # dimension of embedding space

# model training
ALPHA = 1e-3            # learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random walk
WINDOW_SIZE = 4         # window size for word2vec
WALK_LEN = 30           # length of a random walk

# negative sampling
NUM_NEG_SAMPLES = 5     # number of negative samples per positive sample

