from wcss import*
#from lwcss import*
from lstm import*
from transformer import*

import random
from collections import Counter


def get_stream(new_stream = True):
    if new_stream:
        stream = ['A'] * 2000 + ['B'] * 1800 + ['C'] * 1600 + ['D'] * 1400 + ['E'] * 1200
        stream += ['F'] * 1000 + ['G'] * 900 + ['H'] * 800 + ['I'] * 700 + ['J'] * 600
        stream += [random.choice('KLMNOPQRSTUVWXYZ') for _ in range(5000)]
        random.shuffle(stream)
        
        with open('test_stream.txt', 'w') as f:
            for item in stream:
                f.write(f"{item}\n")


    with open('test_stream.txt', 'r') as f:
        stream = [line.strip() for line in f]
    return stream


stream_size = 10000
window_size = 1000
num_buckets = 10
k = 5
eval_interval = 1000
items = get_stream(new_stream=False)
vocab = list(set(items))
item2idx = {item: idx for idx, item in enumerate(vocab)}


# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wcss_model = WCSS(k, window_size, num_buckets)
lstm_model = LSTMPredictor(k, window_size, num_buckets, vocab_size=len(vocab), device=device).to(device)
transformer_model = SimpleTransformer(k, window_size, num_buckets, vocab_size=len(vocab), device=device).to(device)

lstm_model.load_state_dict(torch.load('lstm_model.pt', map_location=device))
transformer_model.load_state_dict(torch.load('transformer_model.pt', map_location=device))

# Ground truth window
window = deque()
true_counts = Counter()

eval = False

for i, item in enumerate(items):
    # update models
    item_idx = ord(item) - ord('A')  
    item_idx_tensor = torch.tensor([[item_idx]])  

    wcss_model.update(item)
    lstm_model.update(item_idx_tensor, item)
    transformer_model.update(item_idx_tensor, item)


    # update ground truth
    window.append(item)
    true_counts[item] += 1
    if len(window) > window_size:
        removed = window.popleft()
        true_counts[removed] -= 1
        if true_counts[removed] == 0:
            del true_counts[removed]

    # evaluation
    if eval:
        if (i + 1) % eval_interval == 0:
            #print(f"\nAfter {i+1} items:")
            true_top = true_counts.most_common(k)
            wcss_top = wcss_model.get_top_k()
            lstm_top = lstm_model.get_top_k()
            transformer_top = transformer_model.get_top_k()

            #print("True top-k:        ", true_top)
            #print("Baseline WCSS top: ", wcss_top)
            #print("LSTM model top:    ", lstm_top)
            #print("Transformer top:   ", transformer_top)

            for model_name, top_k in [('WCSS', wcss_top), ('LSTM', lstm_top), ('Transformer', transformer_top)]:
            # for model_name, top_k in [('WCSS', wcss_top), ('LSTM', lstm_top)]:
                for est_item, est_count in top_k:
                    true_count = true_counts.get(est_item, 0)
                    #print(f"  {model_name} - Item {est_item}: Estimated={est_count}, True={true_count}")

print("\n=== Final Results ===")
print("True top-k:        ", true_counts.most_common(k))
print("Baseline WCSS top: ", wcss_model.get_top_k())
print("LSTM model top:    ", lstm_model.get_top_k())
print("Transformer top:   ", transformer_model.get_top_k())
