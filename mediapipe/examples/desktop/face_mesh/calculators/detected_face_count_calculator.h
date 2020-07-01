
#ifndef MEDIAPIPE_CALCULATORS_FACE_MESH_DETECTED_FACE_COUNT_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_FACE_MESH_DETECTED_FACE_COUNT_CALCULATOR_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {

// A calculator that count input landmarksList size.
//
// Count input landmark(std::vector<NormalizedLandmarkList>) and return this 
// value to ouput_stream. Input IMAGE has no effect on calculation, but is used to
// ensure that the calculator works even when the landmark is empty. And if the 
// input landmark is empty, the number of faces found is zero.
//
// Example config:
// node {
//   calculator: "DetectedFaceCountCalculator"
//   input_stream: "IMAGE:input_image"
//   input_stream: "LANDMARKS:multi_face_landmarks"
//   output_stream: "COUNT:face_count"
// }

constexpr char kInputImageTag[] = "IMAGE";
constexpr char kInputLandmarksTag[] = "LANDMARKS";
constexpr char kOutputCountTag[] = "COUNT";

class DetectedFaceCountCalculator : public CalculatorBase { 
  public:
    DetectedFaceCountCalculator() = default;
    ~DetectedFaceCountCalculator() override = default;

    static ::mediapipe::Status GetContract(CalculatorContract* cc) {
        // Check tag.
        RET_CHECK(cc->Inputs().HasTag(kInputImageTag));
        cc->Inputs().Tag(kInputImageTag).Set<ImageFrame>();
        RET_CHECK(cc->Inputs().HasTag(kInputLandmarksTag));
        cc->Inputs().Tag(kInputLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
        RET_CHECK(cc->Outputs().HasTag(kOutputCountTag));
        cc->Outputs().Tag(kOutputCountTag).Set<int>();

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext* cc) {
        std::unique_ptr<int> face_count;
        if (!cc->Inputs().Tag(kInputLandmarksTag).IsEmpty()) {
            const auto& landmarks = cc->Inputs().Tag(kInputLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
            face_count = absl::make_unique<int>(landmarks.size());
        } else {
            face_count = absl::make_unique<int>(0);
        }
        cc->Outputs().Tag(kOutputCountTag).Add(face_count.release(), cc->InputTimestamp());

        return ::mediapipe::OkStatus();
    };    
};

}

#endif  // MEDIAPIPE_CALCULATORS_FACE_MESH_DETECTED_FACE_COUNT_CALCULATOR_H_