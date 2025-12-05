#include "debug_vis.cuh"
#include "threads/cuda_threads.cuh"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <cstdio>


// -------------------------------- DEBUG VISUALIZATION -------------------------------- //

void debugVis_multistream_side_by_side(
    int num_cameras, 
    int cap_width, 
    int cap_height, 
    const int N,
    cudaStream_t stream2, 
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes,
    ActionVis* d_vis,  // currently unused, but okay to keep
    ActionVis* h_vis
) {
    // 1) Download current frames for ALL cameras from GPU for debug/display
    {
        std::lock_guard<std::mutex> lock2(frame_copy_mutex);

        const int vis_width  = cap_width * num_cameras;
        const int vis_height = cap_height;

        // Prepare per-camera CPU mats
        static std::vector<cv::Mat> cam_frames;
        if (cam_frames.size() != static_cast<size_t>(num_cameras)) {
            cam_frames.assign(num_cameras, cv::Mat());
        }

        for (int cam = 0; cam < num_cameras; ++cam) {
            if (cam_frames[cam].empty() ||
                cam_frames[cam].cols != cap_width ||
                cam_frames[cam].rows != cap_height ||
                cam_frames[cam].type() != CV_8UC3)
            {
                cam_frames[cam].create(cap_height, cap_width, CV_8UC3);
            }

            const unsigned char* src_cam =
                d_bgr_undistorted + static_cast<size_t>(cam) * frame_bytes;

            cudaMemcpyAsync(
                cam_frames[cam].data,
                src_cam,
                frame_bytes,
                cudaMemcpyDeviceToHost,
                stream2
            );
        }

        // Wait for all copies to complete
        cudaStreamSynchronize(stream2);

        // Prepare composite frame_back: [cam0 | cam1 | cam2 | ...]
        if (frame_back.empty() ||
            frame_back.cols != vis_width ||
            frame_back.rows != vis_height ||
            frame_back.type() != CV_8UC3)
        {
            frame_back.create(vis_height, vis_width, CV_8UC3);
        }

        for (int cam = 0; cam < num_cameras; ++cam) {
            cv::Rect roi(cam * cap_width, 0, cap_width, cap_height);
            cam_frames[cam].copyTo(frame_back(roi));
        }

        // 2) Downscale frame_back → frame_front for faster display
        const float debug_scale = 0.25f; // or 0.2 if you want smaller
        cv::resize(
            frame_back,
            frame_front,
            cv::Size(
                static_cast<int>(vis_width  * debug_scale),
                static_cast<int>(vis_height * debug_scale)
            ),
            0, 0, cv::INTER_AREA
        );
    }

    #if !DEBUG_ONLY_FRAME
        // 3) Draw overlays on frame_front (composite)
        static const cv::Scalar COLORS[3] = {
            cv::Scalar(0,255,0),   // green
            cv::Scalar(0,0,255),   // red
            cv::Scalar(0,255,255)  // yellow
        };
        static const char* CLABELS[3] = {"G","R","Y"};

        // frame_back is the full composite; frame_front is scaled version
        float visScaleX = static_cast<float>(frame_front.cols) /
                        static_cast<float>(frame_back.cols);
        float visScaleY = static_cast<float>(frame_front.rows) /
                        static_cast<float>(frame_back.rows);

        auto clampRect = [&](const cv::Rect& r)->cv::Rect {
            return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows);
        };

        {
            std::lock_guard<std::mutex> lock2(frame_copy_mutex);
            for (int i = 0; i < N; ++i) {
                const ActionVis& v = h_vis[i];

                // Skip detections with invalid camera index
                if (v.cam < 0 || v.cam >= num_cameras) {
                    continue;
                }

                // Compute global coordinates in the composite image
                int global_x1 = v.x1 + v.cam * cap_width;
                int global_x2 = v.x2 + v.cam * cap_width;
                int global_y1 = v.y1;
                int global_y2 = v.y2;

                cv::Rect box(
                    cv::Point(
                        static_cast<int>(global_x1 * visScaleX),
                        static_cast<int>(global_y1 * visScaleY)
                    ),
                    cv::Point(
                        static_cast<int>(global_x2 * visScaleX),
                        static_cast<int>(global_y2 * visScaleY)
                    )
                );

                cv::Rect roi = clampRect(box);

                int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                const cv::Scalar col = COLORS[idx];
                cv::rectangle(frame_front, roi, col, 2);

                char txt[64];
                std::snprintf(txt, sizeof(txt), "%s (cam %d)",
                            CLABELS[idx], v.cam);
                cv::putText(
                    frame_front,
                    txt,
                    {roi.x, std::max(0, roi.y - 6)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA
                );
            }
        }
    #endif // !DEBUG_ONLY_FRAME
}


// Single-stream version
void debugVis_single_stream(
    int num_cameras, 
    int cap_width, 
    int cap_height, 
    const int N,
    cudaStream_t stream2, 
    unsigned char* d_bgr_undistorted, 
    size_t frame_bytes,
    ActionVis* d_vis, 
    ActionVis* h_vis
) {

            {

                std::lock_guard<std::mutex> lock2(frame_copy_mutex);

                if (frame_back.empty() ||
                    frame_back.cols != cap_width ||
                    frame_back.rows != cap_height ||
                    frame_back.type() != CV_8UC3)
                {
                    frame_back.create(cap_height, cap_width, CV_8UC3);
                }

                cudaMemcpyAsync(
                    frame_back.data,
                    d_bgr_undistorted,
                    frame_bytes,
                    cudaMemcpyDeviceToHost,
                    stream2
                );
                cudaStreamSynchronize(stream2);

                // 2) Downscale frame_back → frame_front for faster display
                cv::resize(frame_back, frame_front,
                           cv::Size(cap_width / 2, cap_height / 2),
                           0, 0, cv::INTER_AREA);

            }

            // Optional: frame-only debug
            #if DEBUG_ONLY_FRAME
                nvtxRangePop(); // DebugVisualization
                // continue;
            #endif

            // 3) Draw overlays on frame_front
            static const cv::Scalar COLORS[3] = 
                {
                    cv::Scalar(0,255,0),   // green
                    cv::Scalar(0,0,255),   // red
                    cv::Scalar(0,255,255)  // yellow
                };
            static const char* CLABELS[3] = {"G","R","Y"};

            float visScaleX = 0.5f;
            float visScaleY = 0.5f;

            auto clampRect = [&](const cv::Rect& r)->cv::Rect { return r & cv::Rect(0, 0, frame_front.cols, frame_front.rows); };

            {
                std::lock_guard<std::mutex> lock2(frame_copy_mutex);

                for (int i = 0; i < N; ++i) {

                    const ActionVis& v = h_vis[i];

                    cv::Rect box(
                        cv::Point(static_cast<int>(v.x1 * visScaleX),
                                  static_cast<int>(v.y1 * visScaleY)),
                        cv::Point(static_cast<int>(v.x2 * visScaleX),
                                  static_cast<int>(v.y2 * visScaleY))
                    );

                    cv::Rect roi = clampRect(box);

                    int idx = (v.cls < 0) ? 0 : (v.cls > 2 ? 2 : v.cls);
                    const cv::Scalar col = COLORS[idx];
                    cv::rectangle(frame_front, box, col, 2);

                    char txt[64];
                    std::snprintf(txt, sizeof(txt), "%s (cam %d)",
                                  CLABELS[idx], v.cam);
                    cv::putText(frame_front, txt,
                                {box.x, std::max(0, box.y - 6)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv::LINE_AA);

                }

            }

}