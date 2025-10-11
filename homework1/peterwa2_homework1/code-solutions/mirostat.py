def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"
    per_token_ce = []
    ks, s_hats, mus, errors, surprises = [], [], [], [], []

    # TODO: YOUR CODE HERE -- additional variable init
    # We will not be checking this section for correctness,
    # But you will probably eventually want to set up some
    # extra variables here for plotting metrics.
    # Our advice is to fill out the other sections first!

    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)

            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)

        # TODO: YOUR CODE HERE -- Estimate Zipf's exponent
        # Following Basu et al, use m=100 (i.e. use only the top 100 tokens(' diffs) to estimate the exponent)
        # Refer to Equation 30 https://arxiv.org/pdf/2007.14966#equation.C.30 for pointers
        m = min(100, sorted_logits.size(1))
        top = sorted_logits[:, :m]
        top_probs = torch.softmax(top, dim=-1)
        r = torch.arange(1, m+1, device=top_probs.device, dtype=top_probs.dtype)
        x = torch.log(r)
        y = torch.log(top_probs[0])
        x_mean, y_mean = x.mean(), y.mean()
        var_x = ((x - x_mean) ** 2).mean().clamp_min(1e-12)
        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
        slope = cov_xy / var_x

        s_hat = max(1.0001, -slope.item()) # replace with your own expression

        # TODO: YOUR CODE HERE -- Compute k using Zipf exponent
        vocab_size = adjusted_logits.size(-1)
        k_float = ((s_hat - 1.0) * math.exp(mu)) ** (1.0 / s_hat)
        k = int(max(1, min(vocab_size, round(k_float))))

        # top k sampling
        topk_logits = sorted_logits[0:k]
        topk_inds = sorted_inds[0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

        # TODO: YOUR CODE HERE -- Compute surprisal error and adjust mu accordingly
        probs_full = torch.softmax(adjusted_logits, dim=-1)
        p_next = probs_full[0, next_tok.item()].clamp_min(1e-12)
        surprise = -torch.log(p_next)
        per_token_ce.append(surprise.item())
        err = surprise.item() - target_ce
        mu = mu - learning_rate * err

        ks.append(int(k))
        s_hats.append(float(s_hat))
        mus.append(float(mu))
        errors.append(float(err))
        surprises.append(float(surprise))

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    ce_arr = np.array(per_token_ce, dtype=float)
    stats = {
        "mean_ce": float(ce_arr.mean()) if ce_arr.size else float("nan"),
        "median_ce": float(np.median(ce_arr)) if ce_arr.size else float("nan"),
        "std_ce": float(ce_arr.std()) if ce_arr.size else float("nan"),
        "seq_level_perplexity": float(math.exp(ce_arr.mean())) if ce_arr.size else float("nan"),
        # add traces for plotting
        "k_trace": ks,
        "s_hat_trace": s_hats,
        "mu_trace": mus,
        "error_trace": errors,
        "surprisal_trace": surprises,
    }

    return text, stats

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Once upon a time,"
    text, stats = mirostat(model, tokenizer, prompt, max_length=256, device=device, temperature=1.0, target_ce=3.0, learning_rate=0.1)
    print(text)
    print(stats)