//
// Created by user1 on 23/09/2019.
//

#ifndef GST_FACE_DETECTOR_GSTFACEDETECTOROV_H
#define GST_FACE_DETECTOR_GSTFACEDETECTOROV_H

#include <gstreamermm.h>
#include <gstreamermm/private/basetransform_p.h>

#include <memory>
#include "FaceDetectorOpenvino.hpp"
#include "Logger.hpp"
#include "ModelReader.hpp"

using namespace anyvision::anvcv;

constexpr int max_detections = 100;
constexpr float detection_threshold = 0.5;

class GstFaceDetectorOV : public Gst::BaseTransform {
public:
    static void class_init(Gst::ElementClass<GstFaceDetectorOV> *klass) {
        klass->set_metadata("GstFaceDetectorOV",
                            "Analytics", "Openvino Face Detector",
                            "Chris Duncan");

        klass->add_pad_template(
                Gst::PadTemplate::create("sink", Gst::PAD_SINK, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
        klass->add_pad_template(
                Gst::PadTemplate::create("src", Gst::PAD_SRC, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
    }

    explicit GstFaceDetectorOV(GstBaseTransform *gobj)
            : Glib::ObjectBase(typeid(GstFaceDetectorOV)), // type must be registered before use
              Gst::BaseTransform(gobj) {
        set_passthrough(true);
        config.logger = &logger;
        config.model_reader = &model_reader;
        config.network_path = "/home/user1/CLionProjects/GST_Face_Detector/anv_cv_sdk/models/v6.1.0_face_det_IRv5_i255_FP32.xml";
        config.weights_path = "/home/user1/CLionProjects/GST_Face_Detector/anv_cv_sdk/models/v6.1.0_face_det_IRv5_i255_FP32.bin";
        config.core_container = core_container;
        config.device_id = OVInferenceDevice::CPU;
        config.img_width = 1280;
        config.img_height = 720;
        config.min_obj_size = 24;
        face_detector = std::make_unique<FaceDetectorOpenvino>(config);
        bboxes = std::unique_ptr<float[]>{new float[max_detections * size_bbox]};
    }

    static bool register_element(Glib::RefPtr<Gst::Plugin> plugin) {
        return Gst::ElementFactory::register_element(plugin, "facedetectorov", 10,
                                                     Gst::register_mm_type<GstFaceDetectorOV>(
                                                             "facedetectorov"));
    }

    Gst::FlowReturn transform_ip_vfunc(const Glib::RefPtr<Gst::Buffer> &buf) override {
        return Gst::FLOW_OK;
    }

    Gst::FlowReturn
    prepare_output_buffer_vfunc(const Glib::RefPtr<Gst::Buffer> &input, Glib::RefPtr<Gst::Buffer> &buffer) override {
        //Object detect
        auto start = std::chrono::high_resolution_clock::now();
        Gst::MapInfo map;
        input->map(map, Gst::MAP_READ);
        auto mat = cv::Mat(cv::Size(1280, 720), CV_8UC3, map.get_data(), cv::Mat::AUTO_STEP);
        auto num_dets =
                face_detector->infer(mat, 1280, 720, 1280 * 3, bboxes.get())[0];


        buffer = input->create_writable();
        size_t num_dets_thresh = 0;
        for (int i = 0; i < num_dets; ++i) {
            auto bbox = reinterpret_cast<BBox *>(bboxes.get() + i * size_bbox);

            if (bbox->score > detection_threshold) {
                gst_buffer_add_video_region_of_interest_meta(buffer->gobj(), "face", bbox->x1, bbox->y1,
                                                             bbox->x2 - bbox->x1, bbox->y2 - bbox->y1);
                ++num_dets_thresh;
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken to infer: " << duration.count() << " ms\n";
        input->unmap(map);
        return Gst::FLOW_OK;
    }

    ModelReader model_reader;
    anyvision::Logger logger;
    OVCoreContainer core_container;
    DetectorConfigOV config;
    std::unique_ptr<FaceDetectorOpenvino> face_detector = nullptr;
    std::unique_ptr<float[]> bboxes = nullptr;
};


#endif //GST_FACE_DETECTOR_GSTFACEDETECTOROV_H
