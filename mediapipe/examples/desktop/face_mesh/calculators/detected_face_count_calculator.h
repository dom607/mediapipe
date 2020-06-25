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

constexpr char kInputImageTag[] = "IMGAE";
constexpr char kInputLandmarksTag[] = "LANDMARKS";
constexpr char kOutputCountTag[] = "COUNT";

class DetectedFaceCountCalculator : public CalculatorBase { 
  public:
    DetectedFaceCountCalculator() = default;
    ~DetectedFaceCountCalculator() override = default;

    static ::mediapipe::Status GetContract(CalculatorContract* cc) {
        // Check tag.
        RET_CHECK(cc->Inputs().HasTag(kInputImageTag));
        RET_CHECK(cc->Inputs().HasTag(kInputLandmarksTag));
        RET_CHECK(cc->Outputs().HasTag(kOutputCountTag));

        // Set Packet type.
        cc->Inputs().Tag(kInputLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
        cc->Inputs().Tag(kInputImageTag).Set<ImageFrame>();
        cc->Outputs().Tag(kOutputCountTag).Set<int>();

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(CalculatorContext* cc) {
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext* cc) {
        const auto& landmarks = cc->Inputs().Tag(kInputLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();

        if (landmarks.size() == 0) {
            auto output_int = absl::make_unique<int>(0);
            cc->Outputs().Tag(kOutputCountTag).Add(output_int.release(), cc->InputTimestamp());
        } else {
            auto output_int = absl::make_unique<int>(landmarks.size());
            cc->Outputs().Tag(kOutputCountTag).Add(output_int.release(), cc->InputTimestamp());
        }
    };    
}

}

