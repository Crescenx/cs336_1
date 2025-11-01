from cs336_basics.transformer.impl import TransformerLM
from cs336_basics.tokenizer.impl import Tokenizer
from cs336_basics.transformer.attn import softmax 
from cs336_basics.utils.checkpointing import load_checkpoint
import torch
from tqdm.auto import tqdm

def decode(
    prompt: str,
    model: TransformerLM,
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    nucleus_p: float = 1.0,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    initial_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(initial_ids, device=device).unsqueeze(0)

    generated_ids = []

    for _ in tqdm(range(max_tokens - len(initial_ids))):
        with torch.no_grad():
            logits = model(input_ids)
        
        next_token_logits = logits[0, -1, :] / temperature
        if nucleus_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > nucleus_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')

        next_token_probs = softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(next_token_probs, num_samples=1).item()
        if next_token_id in tokenizer.special_token_ids:
            break
        generated_ids.append(next_token_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)

    decoded_text = tokenizer.decode(generated_ids)
    return decoded_text


if __name__ == "__main__":
    from pathlib import Path
    prompt = """
    Pavlidis knew that
"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    vocab_path = PROJECT_ROOT / "data" / "artifacts_tokenizer" / "owt_vocab.pkl"
    merges_path = PROJECT_ROOT / "data" / "artifacts_tokenizer" / "owt_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    src = PROJECT_ROOT / "checkpoints" / "owt" /"latest.pt"
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    )

    _ = load_checkpoint(
        src = src,
        model = model,
        optimizer = None,
    )
    output_text = decode(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_tokens=256,
        temperature=0.75,
        nucleus_p=0.9,
    )


    print(output_text)