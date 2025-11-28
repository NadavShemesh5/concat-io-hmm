import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- JIT Compiled Forward Pass ---
# Moving the loop to a JIT function removes Python interpreter overhead
# and fuses operations, providing a massive speedup for sequential algorithms.
@torch.jit.script
def forward_pass_jit(log_trans: torch.Tensor,
                     emit_scores: torch.Tensor,
                     lengths: torch.Tensor,
                     initial_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Optimized Forward Algorithm Loop
    log_trans: (B, T, K, K)
    emit_scores: (B, T, K) - Pre-gathered emission log probs for observed tokens
    lengths: (B,)
    initial_log_probs: (K,)
    """
    max_len = log_trans.size(1)

    # Initialization (t=0)
    # log_alpha: (B, K)
    log_alpha = initial_log_probs.unsqueeze(0) + emit_scores[:, 0, :]

    # Recursion (t=1 to T-1)
    for t in range(1, max_len):
        # Transition Step
        # log_alpha (prev): (B, K, 1) -> Broadcast over 'To' state
        # log_trans[t]:     (B, K, K) -> (B, From, To)
        # Sum over 'From' dim (dim 1)
        prev_alpha_trans = log_alpha.unsqueeze(2) + log_trans[:, t]
        score_trans = torch.logsumexp(prev_alpha_trans, dim=1) # (B, K)

        # Emission Step
        new_log_alpha = score_trans + emit_scores[:, t, :]

        # Masking: if t >= length, copy old alpha (freeze state)
        # This prevents padding from affecting the sum
        mask = (t < lengths).unsqueeze(1) # (B, 1)
        log_alpha = torch.where(mask, new_log_alpha, log_alpha)

    # Termination: Sum over final states
    return torch.logsumexp(log_alpha, dim=1)

class IOHMM(nn.Module):
    def __init__(self, num_states: int, vocab_size: int, embedding_dim: int = 16, hidden_dim: int = 32):
        super(IOHMM, self).__init__()
        self.num_states = num_states
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.transition_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states * num_states)
        )

        self.emission_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states * vocab_size)
        )

        self.initial_logits = nn.Parameter(torch.randn(num_states))

    def get_matrices(self, inputs: torch.Tensor):
        batch_size, seq_len = inputs.shape
        embedded = self.embedding(inputs)

        trans_logits = self.transition_net(embedded)
        trans_logits = trans_logits.view(batch_size, seq_len, self.num_states, self.num_states)
        log_trans_mat = torch.log_softmax(trans_logits, dim=3)

        emit_logits = self.emission_net(embedded)
        emit_logits = emit_logits.view(batch_size, seq_len, self.num_states, self.vocab_size)
        log_emit_mat = torch.log_softmax(emit_logits, dim=3)

        return log_trans_mat, log_emit_mat

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor):
        batch_size, max_len = sequences.shape

        # Shift inputs: u_t = x_{t-1}. t=0 gets a dummy <start> token (0)
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sequences.device)
        inputs = torch.cat([start_token, sequences[:, :-1]], dim=1)
        targets = sequences

        # 1. Compute all matrices at once
        log_trans, log_emit = self.get_matrices(inputs) # trans: (B,T,K,K), emit: (B,T,K,V)

        # 2. Pre-gather emissions for the specific target tokens
        # We want to extract P(x_t | z_t) for the actual observed x_t
        # targets: (B, T) -> expand to (B, T, K, 1) to match (B, T, K, V)
        targets_expanded = targets.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.num_states, -1)
        # Gather along vocab dimension (3) -> result (B, T, K, 1) -> squeeze to (B, T, K)
        emit_scores = torch.gather(log_emit, 3, targets_expanded).squeeze(3)

        # 3. Run JIT-compiled Forward Algorithm
        initial_log_probs = torch.log_softmax(self.initial_logits, dim=0)
        log_likelihoods = forward_pass_jit(log_trans, emit_scores, lengths, initial_log_probs)

        return -log_likelihoods

def collate_batch(batch: List[np.ndarray]):
    tensors = [torch.LongTensor(item) for item in batch]
    lengths = torch.LongTensor([len(item) for item in tensors])
    padded_seqs = pad_sequence(tensors, batch_first=True, padding_value=0)
    return padded_seqs, lengths

def run_training(
        model: IOHMM,
        train_data: List[np.ndarray],
        valid_data: List[np.ndarray],
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_seqs, batch_lens in tqdm(train_loader):
            batch_seqs, batch_lens = batch_seqs.to(device), batch_lens.to(device)

            optimizer.zero_grad()
            nll_vector = model(batch_seqs, batch_lens)
            loss = nll_vector.mean()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_seqs.size(0)

        model.eval()
        total_nll_sum = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_seqs, batch_lens in valid_loader:
                batch_seqs, batch_lens = batch_seqs.to(device), batch_lens.to(device)
                nll_vector = model(batch_seqs, batch_lens)
                total_nll_sum += nll_vector.sum().item()
                total_tokens += batch_lens.sum().item()

        perplexity = np.exp(total_nll_sum / total_tokens if total_tokens > 0 else 0)

        print(f"Epoch {epoch+1}/{epochs} | Valid Perplexity: {perplexity:.4f}")
