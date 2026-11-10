// fllama_embed: one pooled embedding per call, for retrieval (bge-m3 and
// other encoder models). It does not go through server_context: an embedding
// model needs a context made for embeddings (non-causal attention, pooling),
// which the chat path does not share.

#include "fllama_embed.h"
#include "fllama_backends.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.cpp/common/common.h"
#include "llama.cpp/include/llama.h"

namespace {

// The loaded model and its context, kept for the next call.
struct embed_state {
  std::string model_path;
  int context_size = 0;
  int num_gpu_layers = 0;
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;

  void unload() {
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    ctx = nullptr;
    model = nullptr;
    model_path.clear();
  }
};

std::mutex g_embed_mutex;
// Leaked on purpose, as fllama.cpp's ServerManager: no static destruction
// after the Metal backend is gone.
embed_state &g_embed = *new embed_state();

// Loads [request]'s model and context unless they are the loaded ones.
// Returns 0 or an FLLAMA_EMBED_ERROR_ value.
int32_t ensure_loaded(const fllama_embed_request &request) {
  const std::string path = request.model_path ? request.model_path : "";
  if (g_embed.model && g_embed.ctx && g_embed.model_path == path &&
      g_embed.context_size == request.context_size &&
      g_embed.num_gpu_layers == request.num_gpu_layers) {
    return 0;
  }
  g_embed.unload();

  fllama_backends_init_once();
  auto registry_lock = fllama_backends_registry_read_lock();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = request.num_gpu_layers;
  llama_model *model = llama_model_load_from_file(path.c_str(), mparams);
  if (!model) return FLLAMA_EMBED_ERROR_MODEL_LOAD;

  int n_ctx = request.context_size;
  if (n_ctx <= 0) n_ctx = llama_model_n_ctx_train(model);
  if (n_ctx <= 0) n_ctx = 512;

  llama_context_params cparams = llama_context_default_params();
  cparams.embeddings = true;
  cparams.n_ctx = n_ctx;
  // A non-causal model sees the whole input in one micro-batch.
  cparams.n_batch = n_ctx;
  cparams.n_ubatch = n_ctx;
  cparams.n_seq_max = 1;
  llama_context *ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    llama_model_free(model);
    return FLLAMA_EMBED_ERROR_CONTEXT;
  }
  if (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE) {
    llama_free(ctx);
    llama_model_free(model);
    return FLLAMA_EMBED_ERROR_NOT_POOLED;
  }

  g_embed.model_path = path;
  g_embed.context_size = request.context_size;
  g_embed.num_gpu_layers = request.num_gpu_layers;
  g_embed.model = model;
  g_embed.ctx = ctx;
  return 0;
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int32_t
fllama_embed(struct fllama_embed_request request, float *out,
             int32_t out_len) {
  std::lock_guard<std::mutex> lock(g_embed_mutex);
  // Leaves out the lines that a model load logs, on this thread only.
  fllama_thread_log_scope silence(
      [](enum ggml_log_level, const char *) { return true; });

  const int32_t loaded = ensure_loaded(request);
  if (loaded != 0) return loaded;
  llama_model *model = g_embed.model;
  llama_context *ctx = g_embed.ctx;
  const llama_vocab *vocab = llama_model_get_vocab(model);

  std::vector<llama_token> tokens;
  try {
    tokens = common_tokenize(vocab, request.input ? request.input : "",
                             /* add_special */ true,
                             /* parse_special */ false);
  } catch (...) {
    return FLLAMA_EMBED_ERROR_TOKENIZE;
  }
  if (tokens.empty()) return FLLAMA_EMBED_ERROR_TOKENIZE;
  const size_t n_ctx = llama_n_ctx(ctx);
  if (tokens.size() > n_ctx) {
    // Keep the first tokens, and the end-of-sequence token the model adds.
    const llama_token last = tokens.back();
    tokens.resize(n_ctx);
    if (llama_vocab_get_add_eos(vocab) || llama_vocab_get_add_sep(vocab)) {
      tokens.back() = last;
    }
  }

  llama_memory_clear(llama_get_memory(ctx), true);
  llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
  for (size_t i = 0; i < tokens.size(); i++) {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }
  batch.n_tokens = static_cast<int32_t>(tokens.size());

  int32_t rc;
  if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
    rc = llama_encode(ctx, batch);
  } else {
    rc = llama_decode(ctx, batch);
  }
  llama_batch_free(batch);
  if (rc != 0) return FLLAMA_EMBED_ERROR_DECODE;

  const float *embd = llama_get_embeddings_seq(ctx, 0);
  if (!embd) return FLLAMA_EMBED_ERROR_NOT_POOLED;
  const int32_t n_embd = llama_model_n_embd_out(model);

  double sum = 0;
  for (int32_t i = 0; i < n_embd; i++) sum += double(embd[i]) * embd[i];
  const double norm = sum > 0 ? std::sqrt(sum) : 1.0;
  if (out) {
    const int32_t n = std::min(n_embd, out_len);
    for (int32_t i = 0; i < n; i++) out[i] = static_cast<float>(embd[i] / norm);
  }
  return n_embd;
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_embed_unload(void) {
  std::lock_guard<std::mutex> lock(g_embed_mutex);
  g_embed.unload();
}

} // extern "C"
