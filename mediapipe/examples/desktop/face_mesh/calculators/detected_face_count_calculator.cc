#include "mediapipe/examples/desktop/face_mesh/calculators/detected_face_count_calculator.h"

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

REGISTER_CALCULATOR(DetectedFaceCountCalculator);
}


