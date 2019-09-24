cmake_minimum_required(VERSION 3.14)
project(GST_Face_Detector)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/anv_cv_sdk/cmake/modules/)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenVino)
find_package(GStreamermm)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV 3)

include_directories(. ${GSTREAMERMM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ../utils)
link_directories(${CUDA_LIBRARY_DIRS})

include_directories(anv_cv_sdk/detectors/interface anv_cv_sdk/detectors/face_detector_openvino anv_cv_sdk/utils)
include_directories(anv_cv_sdk/landmarks/interface anv_cv_sdk/landmarks/landmark_detector_openvino)

add_subdirectory(anv_cv_sdk/detectors/face_detector_openvino)
add_subdirectory(anv_cv_sdk/landmarks/landmark_detector_openvino)

add_executable(GST_Face_Detector main.cpp GstFaceDetectorOV.h Draw.cpp Draw.h anv_cv_sdk/utils/Logger.cpp GstLandmarkDetectorOV.h GstFaceDetectorOV.h)
target_link_libraries(GST_Face_Detector ${GSTREAMERMM_LIBRARIES} facedetectopenvino landmarkdetectopenvino ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} )