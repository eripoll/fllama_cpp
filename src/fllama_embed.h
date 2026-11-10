#ifndef FLLAMA_EMBED_H
#define FLLAMA_EMBED_H

#include <stdint.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct fllama_embed_request {
  char *input;        // Required: the text to embed.
  char *model_path;   // Required: an embedding model GGUF whose pooling is not
                      // NONE, such as bge-m3 (CLS pooling).
  int context_size;   // Required: the most tokens embedded. A longer input is
                      // cut to its first tokens. <= 0: the model's trained
                      // context.
  int num_gpu_layers; // Required: -1 for auto, 0 for CPU only, N for N layers.
};

// fllama_embed returns these when it fails.
#define FLLAMA_EMBED_ERROR_MODEL_LOAD -1
#define FLLAMA_EMBED_ERROR_CONTEXT -2
#define FLLAMA_EMBED_ERROR_TOKENIZE -3
#define FLLAMA_EMBED_ERROR_DECODE -4
#define FLLAMA_EMBED_ERROR_NOT_POOLED -5

// Embeds [request.input] with the model's own pooling, normalized to unit
// length (L2), and writes the first [out_len] values to [out]. Returns the
// embedding's dimension count, which may be more than [out_len], or one of
// the FLLAMA_EMBED_ERROR_ values.
//
// Blocks the calling thread. The model and its context stay loaded for the
// next call with the same model path, context size and GPU layers; calls are
// serialized. Text in, floats out: the strings stay the caller's.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int32_t
fllama_embed(struct fllama_embed_request request, float *out, int32_t out_len);

// Frees the model that fllama_embed keeps loaded, if any.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_embed_unload(void);

#ifdef __cplusplus
}
#endif

#endif // FLLAMA_EMBED_H
