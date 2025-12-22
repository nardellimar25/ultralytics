// threads/debug_snapshot.cuh
#pragma once
#include <atomic>
#include <cuda_runtime.h>
#include "cuda_structs.cuh"   // ActionVis

// Configure a fixed maximum (must be <= CLS_MAX_BATCH)
#ifndef DEBUG_SNAPSHOT_MAX_DETS
#define DEBUG_SNAPSHOT_MAX_DETS 64
#endif

// One-time init (call from main or inference thread once)
void debug_snapshot_init(ActionVis* h_vis_from_main /* pinned */, int max_dets = DEBUG_SNAPSHOT_MAX_DETS);

// Producer: called by inference after d_vis is written
void debug_snapshot_publish_async(const ActionVis* d_vis, int N, cudaStream_t stream);

// Consumer: called by debug thread once per displayed frame (non-blocking)
// Returns true if it updated your cached detections.
bool debug_snapshot_try_update_cache();
