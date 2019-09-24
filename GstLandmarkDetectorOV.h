#pragma once
#include <gstreamermm.h>
#include <gstreamermm/private/basetransform_p.h>

#include <memory>
#include <anv_cv_sdk/detectors/interface/DetectorConfigOV.hpp>
#include <anv_cv_sdk/landmarks/landmark_detector_openvino/landmark_detector_ov_utils.hpp>
#include "LandmarkDetectorOpenvino.hpp"
#include "Logger.hpp"
#include "ModelReader.hpp"

using namespace anyvision::anvcv;

constexpr int max_detections_lm = 100;


class GstLandmarkDetectorOV : public Gst::BaseTransform {
public:
    static void class_init(Gst::ElementClass<GstLandmarkDetectorOV> *klass) {
        klass->set_metadata("GstLandmarkDetectorOV",
                            "Analytics", "Openvino Landmark Detector",
                            "Chris Duncan");

        klass->add_pad_template(
                Gst::PadTemplate::create("sink", Gst::PAD_SINK, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
        klass->add_pad_template(
                Gst::PadTemplate::create("src", Gst::PAD_SRC, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
    }

    explicit GstLandmarkDetectorOV(GstBaseTransform *gobj)
            : Glib::ObjectBase(typeid(GstLandmarkDetectorOV)), // type must be registered before use
              Gst::BaseTransform(gobj) {
        set_passthrough(true);
        config.model_reader = &model_reader;
        config.logger = &logger;
        config.network_path = "/home/user1/CLionProjects/GST_Face_Detector/anv_cv_sdk/models/face_landmarks_FP32.xml";
        config.weights_path = "/home/user1/CLionProjects/GST_Face_Detector/anv_cv_sdk/models/face_landmarks_FP32.bin";
        config.core_container = core_container;
        config.device_id = OVInferenceDevice::CPU;
        config.max_batch_size = 1;
        landmark_detector = std::make_unique<LandMarkDetectorOpenvino>(config);
        bboxes = std::unique_ptr<float[]>{new float[max_detections * size_bbox]};
        landmarks = std::unique_ptr<float[]>(new float[size_landmarks * 100]);
        ld_crops = std::unique_ptr<uint8_t []>{new uint8_t[LandMarkDetectorOpenvino::size_crop * max_detections]};
    }

    static bool register_element(Glib::RefPtr<Gst::Plugin> plugin) {
        return Gst::ElementFactory::register_element(plugin, "landmarkdetectorov", 10,
                                                     Gst::register_mm_type<GstLandmarkDetectorOV>(
                                                             "landmarkdetectorov"));
    }

    Gst::FlowReturn transform_ip_vfunc(const Glib::RefPtr<Gst::Buffer> &buf) override {
        return Gst::FLOW_OK;
    }

    Gst::FlowReturn
    prepare_output_buffer_vfunc(const Glib::RefPtr<Gst::Buffer> &input, Glib::RefPtr<Gst::Buffer> &buffer) override {
        //Object detect
        auto start = std::chrono::high_resolution_clock::now();
        Gst::MapInfo map;
        buffer = input;
        input->map(map, Gst::MAP_READ);
        auto addr = map.get_data();
        auto mat = cv::Mat(cv::Size(1280, 720), CV_8UC3, addr, cv::Mat::AUTO_STEP);
        extract_bboxes(input);
        generate_crops_from_image(mat, bboxes.get(), num_dets, ld_crops.get());
        landmark_detector->infer(ld_crops.get(), num_dets, landmarks.get());
        std::unique_ptr<float[]> rescaled_landmarks = rescale_landmarks(landmarks.get(), bboxes.get(), num_dets, 1280, 720);
        buffer = input->create_writable();
        for (int i = 0; i < num_dets; ++i) {
            for (size_t lm_idx = 0; lm_idx < size_landmarks; lm_idx += 2) {
                auto x = rescaled_landmarks[i * size_landmarks + lm_idx];
                auto y = rescaled_landmarks[i * size_landmarks + lm_idx + 1];

                cv::circle(mat, cv::Point(x, y), 3, cv::Scalar(0, 255, 0));
            }
        }


        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken to infer landmarks: " << duration.count() << " ms\n";
        return Gst::FLOW_OK;
    }

    void convert_meta_to_bbox(GstVideoRegionOfInterestMeta* meta, size_t bbox_idx){
        auto bbox = reinterpret_cast<BBox *>(bboxes.get() + bbox_idx * size_bbox);
        bbox->x1 = meta->x;
        bbox->y1 = meta->y;
        bbox->y2 = meta->h + meta->y;
        bbox->x2 = meta->w + meta->x;

    }
    void extract_bboxes(const Glib::RefPtr<Gst::Buffer>& buf) {
        num_dets = 0;
        gpointer state = NULL;
        Gst::MapInfo map;
        size_t meta_idx = 0;
        while ((meta = gst_buffer_iterate_meta(buf->gobj(), &state)) != NULL) {
            if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE)
                continue;
            auto roi_meta = reinterpret_cast<GstVideoRegionOfInterestMeta*>(meta);
            convert_meta_to_bbox(roi_meta, meta_idx++);
        }
        num_dets = meta_idx;
    }

    GstMeta *meta = nullptr;
    ModelReader model_reader;
    anyvision::Logger logger;
    OVCoreContainer core_container;
    LandmarkDetectorConfigOV config;
    std::unique_ptr<LandMarkDetectorOpenvino> landmark_detector = nullptr;
    size_t num_dets = 0;
    std::unique_ptr<uint8_t[]> ld_crops;
    std::unique_ptr<float[]> landmarks;
    std::unique_ptr<float[]> bboxes = nullptr;
};


