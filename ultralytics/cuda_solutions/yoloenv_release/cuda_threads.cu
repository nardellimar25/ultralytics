#include "cuda_threads.cuh"

#include <nvtx3/nvToolsExt.h>
#include "cuda_preprocess.cuh"
#include "cuda_postprocess.cuh"

// ---- define the shared globals declared in the header ----
cv::Mat frame_back;
cv::Mat frame_front;

std::mutex frame_mutex;
std::mutex frame_copy_mutex;
// std::mutex detection_mutex;

std::condition_variable frame_ready;
std::atomic<bool> new_frame_available(false);
std::atomic<bool> keep_running(true);

// ---- implementations  ----

// Thread function to capture frames from webcam
void frame_capture_thread(cv::VideoCapture& cap) {

    nvtxRangePush("FrameCaptureThread");

    while (keep_running) {

        // Event: frame capture
        nvtxRangePush("Frame capture");

        cap.read(frame_back);
        if (frame_back.empty()) {
            std::cerr << "Empty frame\n";
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            new_frame_available = true;
        }

        nvtxRangePop(); // "Frame capture"
        // Notify the inference thread that a new frame is available
        frame_ready.notify_one();

    }
    nvtxRangePop(); // "FrameCaptureThread"
}

// Thread function to run inference
void inference_thread(
    float* engine_input, unsigned char* pinned_frame_data, unsigned char* d_bgr, unsigned char* d_resized,
    int engine_img_width, int engine_img_height, int cap_width, int cap_height,
    float scaleX, float scaleY, size_t pinned_size, cudaStream_t stream1, 
    cudaStream_t stream2,
    nvinfer1::IExecutionContext* context, void** buffers,
    float* engine_output, int num_anchors, float conf_thresh, float iou_thresh,
    Detection* d_final, Detection* d_dets, Detection* d_compacted, int* d_mask, 
    int* d_count_final, int* h_count_final
) {

    nvtxRangePush("InferenceThread");

    while (keep_running) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_ready.wait(lock, [] { return new_frame_available.load(); });
        new_frame_available = false;

        // Event: preprocess
        nvtxRangePush("Preprocessing");
        run_preprocess_gpu( 
            engine_input, pinned_frame_data, d_bgr, d_resized, 
            engine_img_width, engine_img_height, cap_width, cap_height, 
            scaleX, scaleY, pinned_size, stream1
        );
        nvtxRangePop();

        // Event: inference
        nvtxRangePush("Inference");
        context->enqueueV2(buffers, stream1, nullptr);
        nvtxRangePop();

        // Event: post-process
        nvtxRangePush("Postprocessing");
        run_postprocess_gpu(
            engine_output, num_anchors, conf_thresh, iou_thresh,                        
            d_final, stream2, d_dets, d_compacted, 
            d_mask, d_count_final, h_count_final
        );
        nvtxRangePop();

        // Event: Debugging
        nvtxRangePush("Debugging");
        // Only do memory transfer if we have final detections
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        if (*h_count_final > 0) {

            std::vector<Detection> detections(*h_count_final);

            cudaMemcpy(detections.data(), d_final, (*h_count_final) * sizeof(Detection), cudaMemcpyDeviceToHost);
            
            std::lock_guard<std::mutex> lock2(frame_copy_mutex);
            frame_back.copyTo(frame_front);

            for (auto& det : detections) {

                int x = static_cast<int>(det.x1 * scaleX);
                int y = static_cast<int>(det.y1 * scaleY);
                int w = static_cast<int>((det.x2 - det.x1) * scaleX);
                int h = static_cast<int>((det.y2 - det.y1) * scaleY);

                cv::Rect box(cv::Point(x, y), cv::Size(w, h));

                cv::rectangle(frame_front, box, {0, 255, 0}, 2);
                cv::putText(frame_front, "cls " + std::to_string(det.class_id) + " " + std::to_string(det.score),
                            box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0});
            }
        }
        nvtxRangePop(); // "Debugging"
    }
    nvtxRangePop(); // "InferenceThread"
}
