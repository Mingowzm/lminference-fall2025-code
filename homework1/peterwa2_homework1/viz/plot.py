import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

def plot_mirostat_traces(stats):
    steps = range(1, len(stats["k_trace"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()

    axs[0].plot(steps, stats["k_trace"])
    axs[0].set_title("k (dynamic top-k)"); axs[0].set_xlabel("step"); axs[0].set_ylabel("k")

    axs[1].plot(steps, stats["s_hat_trace"])
    axs[1].set_title("Zipf exponent $\hat{s}$"); axs[1].set_xlabel("step"); axs[1].set_ylabel("s_hat")

    axs[2].plot(steps, stats["mu_trace"])
    axs[2].set_title("$\mu$"); axs[2].set_xlabel("step"); axs[2].set_ylabel("mu")

    axs[3].plot(steps, stats["surprisal_trace"], label="surprise")
    axs[3].plot(steps, [stats["surprisal_trace"][0] - stats["error_trace"][0] + (stats["error_trace"][i]) for i in range(len(steps))], alpha=0.3)
    axs[3].set_title("Surprisal"); axs[3].set_xlabel("step"); axs[3].set_ylabel("-log p(next)")

    fig.suptitle("Mirostat traces")
    fig.tight_layout()
    return fig

def plot_logit_distribution(model, tokenizer, prompt, tau, steps=(1, 10, 100), device="cuda"):
    """Plot logit distributions at given generation steps."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    all_logits = {}

    for step in range(1, max(steps)+1):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
        if step in steps:
            all_logits[step] = logits[0].detach().cpu().numpy()

        next_tok = logits.argmax(dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    fig, axs = plt.subplots(1, len(steps), figsize=(15, 4))
    for i, s in enumerate(steps):
        axs[i].plot(sorted(all_logits[s], reverse=True))
        axs[i].set_title(f"Step {s}")
        axs[i].set_xlabel("Rank")
        axs[i].set_ylabel("Logit value")
    fig.suptitle(f"Logit distributions | τ={tau} | prompt={prompt[:20]}...")
    fig.tight_layout()
    return fig