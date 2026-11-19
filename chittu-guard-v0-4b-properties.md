# Tulifo/chittu-guard-v0-4b properties

Published properties of the guard checkpoint. Architecture numbers come from `config.json`. Generation defaults come from `generation_config.json`. Training numbers come from the model card and `eval/training_adapter_config.json`.

Repository: https://huggingface.co/Tulifo/chittu-guard-v0-4b

## Identity

| Property | Value |
|---|---|
| Repository | `Tulifo/chittu-guard-v0-4b` |
| Task | Text generation |
| Architecture | `Qwen3ForCausalLM` |
| Base model | `Qwen3-4B-Instruct-2507` (Alibaba Qwen team) |
| Fine-tune by | Tulifo |
| Languages | English and Tamil |
| Licence | Apache-2.0 |
| Parameters | 4,022,468,096 (about 4.0B) |
| Weight dtype | bfloat16 |
| Download size | about 7.5 GB, in two safetensors files |
| Reasoning mode | Non-thinking. It does not write a `think` block |

## Architecture

| Property | Value |
|---|---|
| Layers | 36 |
| Hidden size | 2,560 |
| Attention heads | 32 query, 8 key/value (grouped-query attention) |
| Head size | 128 |
| MLP intermediate size | 9,728 |
| Activation | SiLU |
| Vocabulary | 151,936 tokens |
| Embedding tying | Input embeddings and output layer share weights |
| Position encoding | RoPE, `rope_theta` 5,000,000, no extra scaling |
| Sliding window | Off |
| KV cache | On (`use_cache: true`) |

## Length

| Property | Value |
|---|---|
| Context window | 262,144 tokens |
| Tokenizer limit | 262,144 tokens |
| Fine-tune sequence cap | 768 tokens |
| End tokens | `<|im_end|>` and `<|endoftext|>` |
| Pad token | `<|endoftext|>` |

The 262,144-token window is the Qwen3 ceiling inherited from Alibaba. Tulifo did not change it. Their LoRA lessons were capped at 768 tokens. Input tokens and generated tokens share the same window.

## Default generation settings

| Property | Value |
|---|---|
| Sampling | On |
| Temperature | 0.7 |
| Top-k | 20 |
| Top-p | 0.8 |

The 35-prompt safety check did not use these defaults. It used greedy decoding, which always picks the single most likely token.

## Fine-tune

| Property | Value |
|---|---|
| Method | LoRA, then merged into the full bf16 weights |
| LoRA rank | 8 |
| LoRA scale | 20 |
| LoRA dropout | 0 |
| Layers adapted | 8 |
| Optimizer | Adam |
| Learning rate | 0.00003 |
| Batch size | 1 |
| Seed | 803 |
| Prompt masking | Off |
| Training data | 5,250 synthetic examples, no real child text |
| Behavior | Short reply, or one of `CRISIS`, `UNSAFE`, `PII`, `SPINE` |

## Routing tokens

| Token | Meaning |
|---|---|
| `CRISIS` | Any hint of self-harm, abuse, or danger |
| `UNSAFE` | Sexual, violent, drug, weapon, or other adult content, in any framing |
| `PII` | Personal information shared or requested |
| `SPINE` | Love, feelings, secrets, or "are you real" — honesty about being a program |

The caller is expected to replace these tokens with scripted, human-written replies. The model does not write the crisis text itself.

## Held-out check (27 Sep 2026)

35 prompts, English, Tamil script, and Tanglish, greedy decoding:

| Category | Score |
|---|---|
| `CRISIS` | 4/8 |
| `UNSAFE` | 8/10 |
| `PII` | 3/3 |
| `SPINE` | 3/3 |
| Safe turns that must be answered, not escalated | 10/11 |
