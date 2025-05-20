import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random

class SimpleTransformer(nn.Module):
    def __init__(self, k, window_size, num_buckets, vocab_size=20, embed_dim=32, num_heads=2, num_layers=1, device='cpu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, 1)

        # WCSS backend
        self.k = k
        self.window_size = window_size
        self.num_buckets = num_buckets
        self.bucket_width = window_size // num_buckets
        self.current_bucket = 0
        self.buckets = [defaultdict(int) for _ in range(num_buckets)]
        self.total_counts = defaultdict(int)
        self.items_seen = 0

        self.device = device
        self.eval()

    def forward(self, x_seq):
        emb = self.embedding(x_seq)  # (batch, seq_len, embed_dim)
        out = self.transformer(emb)
        logits = self.fc(out.mean(dim=1))
        return torch.sigmoid(logits)

    def predict(self, item_idx_tensor):
        with torch.no_grad():
            prob = self.forward(item_idx_tensor).item()
            #print(item_idx_tensor, prob)
            return prob >= 0.67

    def update(self, item_idx_tensor, item_label):
        will_reappear = self.predict(item_idx_tensor)

        if will_reappear:
            self.items_seen += 1
            self.buckets[self.current_bucket][item_label] += 1
            self.total_counts[item_label] += 1

            if self.items_seen % self.bucket_width == 0:
                self._expire_bucket()

            if len(self.total_counts) > self.k:
                min_item = min(self.total_counts, key=self.total_counts.get)
                del self.total_counts[min_item]
                for bucket in self.buckets:
                    if min_item in bucket:
                        del bucket[min_item]

    def _expire_bucket(self):
        expire_bucket_idx = (self.current_bucket + 1) % self.num_buckets
        expired_counts = self.buckets[expire_bucket_idx]
        for item, count in expired_counts.items():
            self.total_counts[item] -= count
            if self.total_counts[item] <= 0:
                del self.total_counts[item]
        self.buckets[expire_bucket_idx].clear()
        self.current_bucket = expire_bucket_idx

    def get_top_k(self):
        return sorted(((item, count + 2) for item, count in self.total_counts.items()), key=lambda x: -x[1])

# === optional training script ===

def get_stream(stream_size=20000):
    stream = ['A'] * 2000 + ['B'] * 1800 + ['C'] * 1600 + ['D'] * 1400 + ['E'] * 1200
    stream += ['F'] * 1000 + ['G'] * 900 + ['H'] * 800 + ['I'] * 700 + ['J'] * 600
    stream += [random.choice('KLMNOPQRSTUVWXYZ') for _ in range(5000)]
    random.shuffle(stream)
    return stream

def generate_training_data(stream, vocab, seq_len=10, window_size=100, num_samples=5000):
    item2idx = {item: idx for idx, item in enumerate(vocab)}
    X = []
    y = []
    for _ in range(num_samples):
        pos = random.randint(seq_len, len(stream) - window_size - 1)
        seq = stream[pos - seq_len:pos]
        target_item = stream[pos]
        future_window = stream[pos + 1: pos + 1 + window_size]
        label = 1 if target_item in future_window else 0
        X.append([item2idx[c] for c in seq])
        y.append(label)
    return torch.tensor(X), torch.tensor(y, dtype=torch.float32)

def train_predictor(model, X_train, y_train, epochs=10, batch_size=64, lr=0.001, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    model.eval()

if __name__ == '__main__':
    stream = get_stream()
    vocab = list(set(stream))
    X_train, y_train = generate_training_data(stream, vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(k=5, window_size=1000, num_buckets=10, vocab_size=len(vocab), device=device).to(device)
    train_predictor(model, X_train, y_train, epochs=10, device=device)
    torch.save(model.state_dict(), 'transformer_model.pt')
