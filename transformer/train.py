
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import model as m
import einops

def train(model, dataloader, optimizer, device, num_epochs=3):
    """
    Train function for a custom model.

    Args:
    - model: Your custom PyTorch model.
    - dataloader: DataLoader providing input and target data.
    - optimizer: Optimizer for updating model parameters.
    - device: Device to run the training (e.g., 'cuda' or 'cpu').
    - num_epochs: Number of training epochs.

    Returns:
    - None
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_ids = einops.rearrange(batch['input_ids'].to(device), 'b s -> b s 1')
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, input_ids) 
            loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":

    model = m.Transformer()
    num_samples = 100
    seq_length = 10
    vocab_size = 3

    model = m.Transformer(
            vocab_size = vocab_size,
            seq_len = seq_length
            )

    data = [
             {
                "input_ids": torch.randint(0, vocab_size, (seq_length,), dtype=torch.float),
                "labels": torch.randint(0, vocab_size, (seq_length,), dtype=torch.long)
             } for _ in range(num_samples)
           ] 

    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    dataloader = DataLoader(data, batch_size=8)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = Adam(model.parameters(), lr=0.001)
    train(model, dataloader, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
