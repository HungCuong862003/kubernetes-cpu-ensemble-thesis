# This file shows the REPLACEMENT for load_timesfm() only.
# Do NOT execute — it's for reference.
def load_timesfm(device="cuda"):
    """TimesFM 2.5 loader using manual safetensors bypass.

    Both from_pretrained() and load_checkpoint() are broken in the
    installed timesfm git HEAD as of Apr 24 2026:
      - from_pretrained: huggingface_hub passes unexpected 'proxies' kwarg
      - load_checkpoint: raise NotImplementedError()

    Workaround: download safetensors via hf_hub_download, load directly
    via model.model.load_state_dict(). Verified: missing_keys=0, unexpected=0.
    """
    import torch
    import timesfm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    log(f"loading TimesFM 2.5 200M on {device} (manual load bypass)...")
    torch.set_float32_matmul_precision("high")

    log("  downloading/locating checkpoint...")
    ckpt_path = hf_hub_download(
        repo_id="google/timesfm-2.5-200m-pytorch",
        filename="model.safetensors",
    )

    log("  instantiating model...")
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)

    log("  loading state_dict...")
    state_dict = load_file(ckpt_path)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"  WARN: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        log("  state_dict loaded cleanly (missing=0 unexpected=0)")

    log(f"  moving to {device}...")
    model.model.to(device)

    log("  compiling ForecastConfig...")
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    log("  TimesFM loaded")
    return model
