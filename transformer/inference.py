import torch
import model as m
from tokenizers import Tokenizer
import einops

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate text continuation from a prompt.
    
    Args:
    - model: Trained transformer model
    - tokenizer: Trained tokenizer
    - prompt: String prompt to continue from
    - max_length: Maximum number of tokens to generate
    - temperature: Controls randomness (lower = more deterministic)
    - device: Device to run inference on
    
    Returns:
    - Generated text string
    """
    model.eval()
    model.to(device)
    
    # Encode the prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long)  # Changed to long (integer)
    
    # Generate tokens one by one
    generated_ids = [int(id) for id in input_ids.tolist()]  # Convert to integer list
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input sequence
            sequence_length = 9
            if len(generated_ids) > sequence_length:
                curr_input = generated_ids[-sequence_length:]
            else:
                curr_input = generated_ids
                
            # Convert to tensor and reshape
            curr_input = torch.tensor(curr_input, dtype=torch.long).to(device)  # Changed to long
            curr_input = einops.rearrange(curr_input, 's -> 1 s')  # Add batch dimension
            curr_input = curr_input.float()  # Convert to float for model processing
            curr_input = einops.rearrange(curr_input, 'b s -> b s 1')  # Add feature dimension

            print(curr_input.shape)

            # Get model predictions
            logits = model(curr_input, curr_input)
            logits = logits[0, -1, :] / temperature  # Get last token predictions
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Stop if we generate an EOS token
            if next_token == tokenizer.token_to_id("[EOS]"):
                break
    
    # Decode the generated tokens
    decoded = tokenizer.decode(generated_ids)
    return decoded

if __name__ == "__main__":
    # Load the saved model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model with the same parameters used during training
    model = m.Transformer(
        vocab_size=8000,
        sequence_length=9,  # sequence_length-1 as used in training
        hidden_size=64,
        attention_size=64,
        feedforward_size=128,
        number_heads=8,
        number_layers=6,
        dropout=0.1
    )
    
    # Load the saved state
    model.load_state_dict(torch.load('transformer_model.pth'))
    tokenizer = Tokenizer.from_file('tokenizer.json')
    
    # Example usage
    prompt = "The quick brown fox is very fast but he"
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=50,
        temperature=0.7
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
