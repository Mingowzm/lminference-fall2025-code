from transformers.generation.logits_process import LogitsProcessor
import torch

class OddSamplingLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)

        # Create a mask for odd and even ranks
        batch_size, _ = scores.shape
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for b in range(batch_size):
            # odd ranks: 0, 2, 4, ... (0-based indexing)
            odd_positions = sorted_indices[b, 0::2]
            mask[b, odd_positions] = True

        # Set logits of even rank to -inf
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores

if __name__ == "__main__":
    logits = torch.tensor([[10.0, 5.0, 3.0, 2.0, 1.0, -1.0]])
    print("Original logits:", logits)

    processor = OddSamplingLogitsProcessor()
    new_logits = processor(None, logits.clone())

    print("Processed logits:", new_logits)