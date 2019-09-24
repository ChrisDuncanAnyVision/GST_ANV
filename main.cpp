#include <iostream>
#include <gstreamermm.h>
#include <glibmm.h>
#include "GstFaceDetectorOV.h"
#include "GstLandmarkDetectorOV.h"
#include "Draw.h"

int main(int argc, char *argv[])
{
    Gst::init();
    Gst::Plugin::register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                                 "facedetectorov", "blah",
                                 sigc::ptr_fun(&GstFaceDetectorOV::register_element),
                                 "0.1.0", "LGPL", "", "", "");
    Gst::Plugin::register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                                 "landmarkdetectorov", "blah",
                                 sigc::ptr_fun(&GstLandmarkDetectorOV::register_element),
                                 "0.1.0", "LGPL", "", "", "");

    Gst::Plugin::register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR,
                                 "draw", "blah",
                                 sigc::ptr_fun(&Draw::register_element),
                                 "0.1.0", "LGPL", "", "", "");

    auto pipeline = Gst::Pipeline::create();
    auto source = Gst::FileSrc::create();
    source->property_location() = "/home/user1/CLionProjects/GST_Face_Detector/anv_cv_sdk/test_data/face/face_test_video.mp4";
    auto decodebin = Gst::ElementFactory::create_element("decodebin");
    auto convert = Gst::ElementFactory::create_element("videoconvert");
    auto convertrender = Gst::ElementFactory::create_element("videoconvert");
    auto sink = Gst::ElementFactory::create_element("fpsdisplaysink");
    sink->property("sync", false);
    auto queue = Gst::Queue::create();
    auto queuelm = Gst::Queue::create();
    auto face_detector_ov = Gst::ElementFactory::create_element("facedetectorov");
    auto landmark_detector_ov = Gst::ElementFactory::create_element("landmarkdetectorov");
    auto filter2 = Gst::ElementFactory::create_element("draw");
    decodebin->signal_pad_added().connect([&](const Glib::RefPtr<Gst::Pad>& pad) {
        Glib::RefPtr<Gst::Caps> caps = pad->get_current_caps();

        auto media_type = caps->get_structure(0).get_name();

        std::cout << "Media type: " << media_type << "\n";
        if (media_type == "video/x-raw"){
            if (pad->link(convert->get_static_pad("sink")) != Gst::PAD_LINK_OK)
            {
                throw std::runtime_error("Cannot link convertor");
            }
        }


    });

    pipeline->add(source)->add(decodebin)->add(convert)->add(sink)->add(queue)->add(face_detector_ov)->add(landmark_detector_ov)->add(filter2)->add(convertrender)->add(queuelm);
    source->link(decodebin);
    convert->link(queue)->link(face_detector_ov, Gst::Caps::create_from_string("video/x-raw, width=1280, height=720, format=BGR"))->link(queuelm)->link(landmark_detector_ov)->link(filter2)->link(convertrender)->link(sink);
    pipeline->set_state(Gst::STATE_PLAYING);

    auto mainloop = Glib::MainLoop::create();
    mainloop->run();

    return 0;
}