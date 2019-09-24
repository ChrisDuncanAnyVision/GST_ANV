//
// Created by user1 on 24/09/2019.
//

#ifndef GST_FACE_DETECTOR_DRAW_H
#define GST_FACE_DETECTOR_DRAW_H


#include <gstreamermm.h>
#include <gstreamermm/private/basetransform_p.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

static inline void draw_bboxes(Glib::RefPtr<Gst::Buffer>& buf) {
    gpointer state = NULL;
    GstMeta *meta = NULL;

    Gst::MapInfo map;
    auto buffer = buf->create_writable();
    buffer->map(map, Gst::MAP_READ);
    auto mat = cv::Mat(cv::Size(1280, 720), CV_8UC3, map.get_data(), cv::Mat::AUTO_STEP);

    while ((meta = gst_buffer_iterate_meta(buf->gobj(), &state)) != NULL) {
        if (meta->info->api != GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE)
            continue;
        auto roi_meta = reinterpret_cast<GstVideoRegionOfInterestMeta*>(meta);
//        printf("Object bounding box %d,%d,%d,%d\n", roi_meta->x, roi_meta->y, roi_meta->w, roi_meta->h);
        cv::rectangle(mat, cvPoint(roi_meta->x, roi_meta->y), cvPoint(roi_meta->x + roi_meta->w, roi_meta->y + roi_meta->h),
                      cv::Scalar(255, 0, 255), 2, 1, 0);
//        for (GList *l = roi_meta->params; l; l = g_list_next(l)) {
//            auto structure = (GstStructure *) l->data;
//            printf("  Attribute %s\n", gst_structure_get_name(structure));
//            if (gst_structure_has_field(structure, "label")) {
//                printf("    label=%s\n", gst_structure_get_string(structure, "label"));
//            }
//            if (gst_structure_has_field(structure, "confidence")) {
//                double confidence;
//                gst_structure_get_double(structure, "confidence", &confidence);
//                printf("    confidence=%.2f\n", confidence);
//            }
//        }
    }
}

class Draw  : public Gst::BaseTransform{
public:
    static void class_init(Gst::ElementClass<Draw> *klass)
    {
        klass->set_metadata("draw_longname",
                            "draw_classification", "draw_detail_description", "draw_detail_author");

        klass->add_pad_template(Gst::PadTemplate::create("sink", Gst::PAD_SINK, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
        klass->add_pad_template(Gst::PadTemplate::create("src", Gst::PAD_SRC, Gst::PAD_ALWAYS, Gst::Caps::create_any()));
    }

    explicit Draw(GstBaseTransform *gobj)
            : Glib::ObjectBase(typeid (Draw)), // type must be registered before use
              Gst::BaseTransform(gobj)
    {
        set_passthrough(true);
    }

    static bool register_element(const Glib::RefPtr<Gst::Plugin>& plugin)
    {
        return Gst::ElementFactory::register_element(plugin, "draw", 10, Gst::register_mm_type<Draw>("draw"));
    }

    Gst::FlowReturn transform_ip_vfunc(const Glib::RefPtr<Gst::Buffer>& buf) override{
        return Gst::FLOW_OK;
    }

    Gst::FlowReturn prepare_output_buffer_vfunc(const Glib::RefPtr<Gst::Buffer>& input, Glib::RefPtr<Gst::Buffer>& buffer) override
    {
        auto start = std::chrono::high_resolution_clock::now();

        buffer = input->create_writable();

        draw_bboxes(buffer);


        return Gst::FLOW_OK;
    }

};


#endif //GST_FACE_DETECTOR_DRAW_H
