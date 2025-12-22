// src/cuda/debug_snapshot.cu
#include "threads/debug_snapshot.cuh"
#include "debug_vis.cuh"      // debugVis_update_cached_detections(...)
#include <algorithm>

static ActionVis* g_h_vis_pp[2] = {nullptr, nullptr};
static cudaEvent_t g_ev_done[2];
static std::atomic<int> g_ready_idx{-1};
static std::atomic<int> g_ready_N{0};
static int g_pp_idx = 0;
static int g_max_dets = DEBUG_SNAPSHOT_MAX_DETS;
static bool g_inited = false;

void debug_snapshot_init(ActionVis* h_vis_from_main, int max_dets) {
    if (g_inited) return;

    g_max_dets = (max_dets > 0) ? max_dets : DEBUG_SNAPSHOT_MAX_DETS;

    // Buffer 0: reuse pinned host buffer from main (you already cudaMallocHost it)
    g_h_vis_pp[0] = h_vis_from_main;

    // Buffer 1: allocate one extra pinned buffer (fixed size)
    cudaMallocHost(&g_h_vis_pp[1], g_max_dets * sizeof(ActionVis));

    cudaEventCreateWithFlags(&g_ev_done[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_ev_done[1], cudaEventDisableTiming);

    g_inited = true;
}

void debug_snapshot_publish_async(const ActionVis* d_vis, int N, cudaStream_t stream) {
    if (!g_inited) return; // or call init here if you prefer

    if (!d_vis || N <= 0) {
        g_ready_N.store(0, std::memory_order_release);
        g_ready_idx.store(-1, std::memory_order_release);
        return;
    }

    int Nclamped = std::min(N, g_max_dets);

    g_pp_idx ^= 1;

    cudaMemcpyAsync(
        g_h_vis_pp[g_pp_idx],
        d_vis,
        Nclamped * sizeof(ActionVis),
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaEventRecord(g_ev_done[g_pp_idx], stream);

    g_ready_N.store(Nclamped, std::memory_order_release);
    g_ready_idx.store(g_pp_idx, std::memory_order_release);
}

bool debug_snapshot_try_update_cache() {
    if (!g_inited) return false;

    int idx = g_ready_idx.load(std::memory_order_acquire);
    int N   = g_ready_N.load(std::memory_order_acquire);

    if (idx < 0 || idx > 1) return false;

    if (N <= 0) {
        // Optional: clear cache if you want immediate "no boxes"
        // debugVis_update_cached_detections(nullptr, 0);
        g_ready_idx.store(-1, std::memory_order_release);
        return false;
    }

    if (cudaEventQuery(g_ev_done[idx]) != cudaSuccess) {
        return false; // not ready yet
    }

    debugVis_update_cached_detections(g_h_vis_pp[idx], N);
    g_ready_idx.store(-1, std::memory_order_release); // consumed
    return true;
}
