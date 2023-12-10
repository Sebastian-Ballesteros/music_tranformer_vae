import os
import torch
from GPTmuseVAE import GPTmuseVAE
from miditok.pytorch_data import DatasetTok
from miditok import REMI
from torchtoolkit.data import create_subsets
from pathlib import Path
import utils as ut
import matplotlib.pyplot as plt
from tqdm import tqdm


# Model Hyperparameters
n_embd = 64
n_head = 8
n_layer = 4
z_dim = 16
block_size = 254 # what is the maximum context length for predictions?
dropout = 0.2
########################

#Training hyperparameters

patience = 10  # Number of consecutive iterations without improvement to tolerate
batch_size = 32 # how many independent sequences will we process in parallel?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 15000
eval_interval = 200
learning_rate = 1e-4
eval_iters = 200

# ------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = subset_train if split == 'train' else subset_valid
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ut.get_batch(data, batch_size)

            logits, pred_loss, loss_vae = model(X, Y)
            loss = pred_loss
            
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

tokenizer = REMI(params= Path('midi_dataset_tokenizer_bpe.conf'))
vocab_size = len(tokenizer)

tokens_paths = list(Path('midi_dataset_tokens_no_bpe').glob("**/*.json"))

dataset = DatasetTok(
    tokens_paths, 
    max_seq_len=block_size+1, # to make target and prediction match the song length of block size
    min_seq_len=block_size+1, 
    one_token_stream=False
)

subset_train, subset_valid = create_subsets(dataset, [0.3])

model = GPTmuseVAE( vocab_size= len(tokenizer),
                    n_embd = n_embd,
                    n_head = n_head,
                    n_layer = n_layer,
                    z_dim = z_dim, 
                    block_size = block_size,
                    dropout = dropout)

m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Folder to save checkpoints
checkpoint_folder = "checkpoints"
os.makedirs(checkpoint_folder, exist_ok=True)

# Lists to store training and validation losses for plotting
train_losses = []
val_losses = []


best_val_loss = float('inf')
counter = 0

for iter in tqdm(range(max_iters)):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss, val_loss = losses['train'], losses['val']
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_{iter}.pt")
        torch.save({
            'iteration': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at iteration {iter}...")
                break

    # Sample a batch of data
    xb, yb = ut.get_batch(subset_train, batch_size)

    # Evaluate the loss
    logits, pred_loss, loss_vae = model(xb, yb)
    loss = pred_loss + loss_vae
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save losses for further analysis or plotting
torch.save({
    'train_losses': train_losses,
    'val_losses': val_losses,
}, os.path.join(checkpoint_folder, 'losses_checkpoint.pt'))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig(os.path.join(checkpoint_folder, 'loss_plot.png'))
plt.show()