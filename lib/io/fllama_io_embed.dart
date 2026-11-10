import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:fllama_cpp/fllama_io.dart';
import 'package:fllama_cpp/fllama_universal.dart';
import 'package:fllama_cpp/io/fllama_bindings_generated.dart';

/// The most dimensions [fllamaEmbed] reads. bge-m3 has 1024.
const int _maxDimensions = 16384;

/// Embeds [request.input]: the model's pooled embedding, normalized to unit
/// length (L2).
///
/// Runs on a short-lived isolate, so it never blocks the calling one. The
/// model stays loaded natively for the next call with the same model path,
/// context size and GPU layers; calls run one at a time.
///
/// Throws a [StateError] if the model does not load, is not an embedding
/// model, or the input is empty.
Future<Float32List> fllamaEmbed(FllamaEmbedRequest request) {
  final String input = request.input;
  final String modelPath = request.modelPath;
  final int contextSize = request.contextSize;
  final int numGpuLayers = request.numGpuLayers;
  return Isolate.run(
    () => _embed(input, modelPath, contextSize, numGpuLayers),
    debugName: 'fllama embed',
  );
}

/// Frees the model that [fllamaEmbed] keeps loaded, if any.
Future<void> fllamaEmbedUnload() => Isolate.run(
  () => fllamaBindings.fllama_embed_unload(),
  debugName: 'fllama embed unload',
);

Float32List _embed(
  String input,
  String modelPath,
  int contextSize,
  int numGpuLayers,
) {
  final Pointer<fllama_embed_request> request = calloc<fllama_embed_request>();
  final Pointer<Float> out = calloc<Float>(_maxDimensions);
  try {
    request.ref.input = input.toNativeUtf8().cast<Char>();
    request.ref.model_path = modelPath.toNativeUtf8().cast<Char>();
    request.ref.context_size = contextSize;
    request.ref.num_gpu_layers = numGpuLayers;
    final int n = fllamaBindings.fllama_embed(request.ref, out, _maxDimensions);
    if (n <= 0) {
      throw StateError('fllamaEmbed: ${_reason(n)} ($modelPath)');
    }
    if (n > _maxDimensions) {
      throw StateError('fllamaEmbed: $n dimensions, more than $_maxDimensions');
    }
    return Float32List.fromList(out.asTypedList(n));
  } finally {
    calloc.free(request.ref.input);
    calloc.free(request.ref.model_path);
    calloc.free(request);
    calloc.free(out);
  }
}

String _reason(int code) => switch (code) {
  FLLAMA_EMBED_ERROR_MODEL_LOAD => 'the model did not load',
  FLLAMA_EMBED_ERROR_CONTEXT => 'the context could not be created',
  FLLAMA_EMBED_ERROR_TOKENIZE => 'the input is empty or cannot be tokenized',
  FLLAMA_EMBED_ERROR_DECODE => 'the input could not be evaluated',
  FLLAMA_EMBED_ERROR_NOT_POOLED => 'the model gives no pooled embedding',
  _ => 'error $code',
};
