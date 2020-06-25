// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "multi_face_landmarks";
constexpr char kOutputFaceCountStream[] = "face_count";
constexpr char kDetectionStream[] = "face_detections";
constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller faceCountPoller,
                   graph.AddOutputStreamPoller(kOutputFaceCountStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    LOG(INFO) << "Routine start!";
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Video is empty!";
      break;  // End of video.
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    LOG(INFO) << "Feed image buffer";
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    mediapipe::Packet faceCountPacket;

    if (!faceCountPoller.Next(&faceCountPacket)) {
      LOG(INFO) << "Polling face count failed!";
      break;
    }

    LOG(INFO) << "Try to polling face count.";
    auto& faceCount = faceCountPacket.Get<int>();

    LOG(INFO) << "Face count : " << faceCount;
    if (faceCount == 0) {
      continue;
    }

    LOG(INFO) << "Try to polling landmarks.";
    if (!poller.Next(&packet)) {
      LOG(INFO) << "Polling lanmark failed!";
      break;
    } 

    LOG(INFO) << "Get landmarks.";
    auto& outputLandmarkLists = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    auto& landmarks = outputLandmarkLists[0];
    int landmarksCount = landmarks.landmark_size();
    LOG(INFO) << "Landmark count : " << landmarksCount;

    // for (int i = 0 ; i < landmarksCount ; i++) 
    // {
    //   auto& landmark = landmarks.landmark(i);
    //   if (landmark.has_x()) { LOG(INFO) << "Index [" <<i << "] x : " <<landmark.x();}
    //   if (landmark.has_y()) { LOG(INFO) << "Index [" <<i << "] y : " <<landmark.y();}
    //   if (landmark.has_z()) { LOG(INFO) << "Index [" <<i << "] z : " <<landmark.z();}
    //   if (landmark.has_visibility()) { LOG(INFO) << "Index [" <<i << "] vis : " <<landmark.visibility();}
    // }
    

    // mediapipe::Packet detectionPacket;
    // if (!detectionPoller.Next(&detectionPacket)) {
    //   LOG(INFO) << "Failed to get detectionPacket!";
    // } else {
    //   auto& outputDetections = detectionPacket.Get<std::vector<mediapipe::Detection>>();

    //   if (!outputDetections.empty()) {
    //     auto& detection = outputDetections[0];
    //     if (detection.has_location_data()) {
    //       auto& locationData = detection.location_data();
          
    //       int keyPointSize = locationData.relative_keypoints_size();

    //       if (keyPointSize != 0) {
    //         for (int i = 0 ; i < keyPointSize ; i++) {
    //           auto& keyPoint = locationData.relative_keypoints(i);
    //           if (keyPoint.has_keypoint_label()) {
    //             LOG(INFO) << keyPoint.keypoint_label();
    //           } else {
    //             LOG(INFO) << "Label is empty.";
    //           }
    //         }
    //       } else {
    //         LOG(INFO) << "keyPoint is empty!";
    //       }
    //     } else {
    //       LOG(INFO) << "Location is empty!";
    //     }
    //   } else {
    //     LOG(INFO) << "Detection is empty!";
    //   }
    // };

    LOG(INFO) << "Complete!";
    // auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // // Convert back to opencv for display or saving.
    // cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    // if (save_video) {
    //   if (!writer.isOpened()) {
    //     LOG(INFO) << "Prepare video writer.";
    //     writer.open(FLAGS_output_video_path,
    //                 mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
    //                 capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
    //     RET_CHECK(writer.isOpened());
    //   }
    //   writer.write(output_frame_mat);
    // } else {
    //   cv::imshow(kWindowName, output_frame_mat);
    //   // Press any key to exit.
    //   const int pressed_key = cv::waitKey(5);
    //   if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    // }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << argv[0];
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
