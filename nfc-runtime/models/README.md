# Local models (not committed)

Preferred (better replies):

- `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1.1 GiB)

```bash
cd nfc-runtime/models
curl -L -o Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

Fallback (weak): `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

CLI: `../tools/llama-b10250/llama-completion`

```bash
./scripts/run-real-chat.sh
# Settings → Import Qwen GGUF → Models → Load → Chat
```
