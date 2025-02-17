
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import transformer.model as m
import einops


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# def train(model, dataloader, optimizer, device, num_epochs=3):
#     """
#     Train function for a custom model.
#
#     Args:
#     - model: Your custom PyTorch model.
#     - dataloader: DataLoader providing input and target data.
#     - optimizer: Optimizer for updating model parameters.
#     - device: Device to run the training (e.g., 'cuda' or 'cpu').
#     - num_epochs: Number of training epochs.
#
#     Returns:
#     - None
#     """
#     model.to(device)
#     model.train()
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for batch in dataloader:
#             input_ids = einops.rearrange(batch['input_ids'].to(device), 'b s -> b s 1')
#             labels = batch['labels'].to(device)
#
#             logits = model(input_ids, input_ids) 
#             loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         avg_loss = epoch_loss / len(dataloader)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
#
#



def create_tokenizer(vocab_size, set_ : str):
    """Create and train a BPE tokenizer on the dataset"""

    if set_ not in ['test', 'train']:
        return 'Set is not "test" or "train"'

    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset('tiny_shakespeare', split=set_)
        
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            vocab_size=vocab_size
        )
        tokenizer.pre_tokenizer = Whitespace()
        
        print("Training tokenizer...")
        return dataset, tokenizer, trainer
        
    except Exception as e:
        print(f"Error creating tokenizer: {str(e)}")
        raise


def create_dataloader(set_ : str, batch_size : int, seq_length : int, vocab_size : int):

    dataset, tokenizer, trainer = create_tokenizer(vocab_size, set_)
    
    def batch_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i:i + 1000]["text"]
    
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    processed_data = []
    for item in dataset:
        processed_data.extend(prepare_data(item, tokenizer, seq_length))

    dataloader = DataLoader(processed_data, 
                            batch_size=batch_size, 
                            shuffle=True)

    return dataloader


def train_one_epoch(model, dataloader, optimizer, device):
    epoch_loss = 0

    for batch in tqdm(dataloader):
        input_ids = einops.rearrange(batch['input_ids'].to(device), 'b s -> b s 1')
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, input_ids) 
        loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    return model, avg_loss


def test_one_epoch(model, dataloader, device):

    accuracy = 0.0
    number_batches = 0

    for batch in tqdm(dataloader):
        input_ids = einops.rearrange(batch['input_ids'].to(device), 'b s -> b s 1')
        labels = batch['labels'].to(device)

        logits = model(input_ids, input_ids)

        model_out = torch.argmax(F.softmax(logits.view(-1, logits.size(-1)), dim=-1), dim=-1)
        label_out = labels.view(-1) 

        acc = torch.sum(model_out == label_out)

        number_batches += 1
        accuracy += acc

    return accuracy/number_batches


def prepare_data(examples, tokenizer, seq_length):

    encoded = tokenizer.encode(examples["text"])
    input_ids = encoded.ids
    
    windows = []
    for i in range(0, len(input_ids) - seq_length, seq_length // 2):
        window = input_ids[i:i + seq_length]
        if len(window) == seq_length:
            windows.append({
                "input_ids": torch.tensor(window[:-1], dtype=torch.float),
                "labels": torch.tensor(window[1:], dtype=torch.long)
            })
    return windows


def train(model, batch_size:int, num_epochs:int):

    seq_length = model.sequence_length
    vocab_size = model.vocab_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = create_dataloader('train', batch_size, seq_length + 1, vocab_size)
    test_set  = create_dataloader('test', batch_size, seq_length + 1, vocab_size)

    optimizer = Adam(model.parameters(), lr=3e-4)

    model.to(device)
    model.train()

    print('---- Training started ----')
    for epoch in range(num_epochs):
        model, avg_loss = train_one_epoch(model, train_set, optimizer, device)
        accuracy = test_one_epoch(model, test_set, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy}")

    torch.save(model.state_dict(), 'transformer_model.pth')
    # tokenizer.save('tokenizer.json')
    return model


# if __name__ == "__main__":

#     num_epochs = 10
#     seq_length = 10
#     vocab_size = 8000
#     batch_size = 32
#     embed_dim = 64
#     hidden_dim = 128
#
#     model = m.Transformer(
#         vocab_size=vocab_size,
#         sequence_length = seq_length-1,
#         hidden_size = embed_dim,
#         attention_size = embed_dim,
#         feedforward_size = hidden_dim,
#         number_heads=8,
#         number_layers=6,
#         dropout=0.1
#     )
#
#     model = train(model, batch_size, num_epochs, seq_length)

