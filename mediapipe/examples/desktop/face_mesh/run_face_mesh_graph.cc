// Library to use face mesh with landmark.

#include <cstdlib>

#include "mediapipe/examples/desktop/face_mesh/run_face_mesh_graph.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputVideoStream[] = "input_video";
constexpr char kOutputNormalizedLandmarksStream[] = "multi_face_landmarks";
constexpr char kOutputFaceCountStream[] = "face_count";

typedef struct InternalData {
  mediapipe::CalculatorGraph graph;
  mediapipe::OutputStreamPoller* faceCountPoller;
  mediapipe::OutputStreamPoller* landmarksPoller;
  std::chrono::system_clock::time_point startTime;
  ~InternalData() {
    if (!faceCountPoller)
      delete faceCountPoller;
    if (!landmarksPoller)
      delete landmarksPoller;
  }
};

// For RunGraph with copy. Media Pipe's image frame basically only 
// provides an interface in the form of taking copy and ownership.
// The following deconstructor is used to provide a way to use 
// only the external buffer.
void customDeleter(uint8_t* data) {
  return;
}

void initGraph(FaceMeshDataStructure** data) 
{
  // Load graph pbtxt.
  // Create graph config.
  // Create graph.
  // Attach poller.
  // Save and return.
  std::string excutablePath = "/Users/june/Documents/PUFFNative/native/build/sample/PuffSample.app/Contents/MacOS/PuffSample";
  google::InitGoogleLogging(excutablePath.c_str());
  LOG(INFO) << "Start graph initialization";

  std::string graphPath = "/Users/june/repositories/mediapipe/mediapipe/graphs/face_mesh/face_mesh_landmark_cpu.pbtxt";
  std::string graphContents;

  *data = nullptr;

  InternalData* internalData = new InternalData();

  // Get graph and prepare config for graph init.
  mediapipe::file::GetContents(graphPath, &graphContents);

  if (graphContents.empty()) {
    std::cout << "Failed to load graph contents from file" << std::endl;
    // LOG(INFO) << "Failed to load graph contents from file";
  } else {
    std::cout << graphContents << std::endl;
  }

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          graphContents);

  // Init graph.
  auto result = internalData->graph.Initialize(config).ok();

  if (!result) {
    std::cout << "Failed to initialize graph" << std::endl;

    // LOG(INFO) << "Failed to initialize graph";
    delete internalData;
    return;
  }
  
  auto landmarksSOP = internalData->graph.AddOutputStreamPoller(kOutputNormalizedLandmarksStream);
  if (!landmarksSOP.ok()) {
    delete internalData;
    return;
  }

  auto faceCountSOP = internalData->graph.AddOutputStreamPoller(kOutputFaceCountStream);
  if (!faceCountSOP.ok()) {
    delete internalData;
    return;
  }
  // Start graph.
  result = internalData->graph.StartRun({}).ok();
  if (!result) {
    delete internalData;
    return;
  }

  // Store result.  
  internalData->landmarksPoller = new mediapipe::OutputStreamPoller(std::move(landmarksSOP).ValueOrDie());
  internalData->faceCountPoller = new mediapipe::OutputStreamPoller(std::move(faceCountSOP).ValueOrDie());
  internalData->startTime = std::chrono::system_clock::now();

  FaceMeshDataStructure* ds = new FaceMeshDataStructure();
  ds->internal = (void*)(internalData);
  *data = ds;
}

// Note.
// Mediapipe support synchronous and asychronous polling. In this library, we use synchronous polling.
void RunGraph(FaceMeshDataStructure* data, const PixelFormat pixelformat, 
    const int width, const int height, const bool useCopy,
    const unsigned char* imageData, int* outFaceCount, float** outLandmarksData) {

  // Prepare data.
  auto internalData = (InternalData*)data->internal;

  *outFaceCount = 0;
  *outLandmarksData = nullptr;

  std::cout << "Prepare complete" << std::endl;

  // Make iamge buffer.
  // Note : There are no pixelformat BGR.
  std::unique_ptr<mediapipe::ImageFrame> imageFrame;
  mediapipe::ImageFormat::Format imageformat;
  int widthStep;

  switch (pixelformat) {
    case SRGB:
      imageformat = mediapipe::ImageFormat::SRGB;
      widthStep = width * 3;
      break;
    case SRGBA:
      imageformat = mediapipe::ImageFormat::SRGBA;
      widthStep = width * 4;
      break;
    case SBGRA:
      imageformat = mediapipe::ImageFormat::SBGRA;
      widthStep = width * 4;
      break;
  }

  std::cout << "Try to make ImageFrame." << std::endl;
  if (useCopy) {
    imageFrame = absl::make_unique<mediapipe::ImageFrame>(imageformat, width, height, 
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);    
    auto pixelData = imageFrame->MutablePixelData();
    memcpy(pixelData, imageData, height * widthStep);
  } else {
    imageFrame = absl::make_unique<mediapipe::ImageFrame>(imageformat, width, height, 
      widthStep, (uint8_t*)imageData, customDeleter);
  }
  std::cout << "ImageFrame creation complete." << std::endl;

  // Get frame timestamp.
  auto pts = (std::chrono::system_clock::now() - internalData->startTime).count();

  // Wrap image with Packet and submit this.
  std::cout << "Feed ImageFrame. pts : " << pts << std::endl;
  mediapipe::Packet p = mediapipe::Adopt(imageFrame.release()).At(mediapipe::Timestamp(pts)); // Add timestamp;
  std::cout << "Add packet" << std::endl;
  internalData->graph.AddPacketToInputStream(kInputVideoStream, p);
  std::cout << "Feed complete" << std::endl;

  // Get result.
  // Note : Below codes works synchronously. Means if graph failed to generate some output,
  //        it will be stuck.

  // Custom designed graphs will always return the number of faces found.

  std::cout << "Try to polling face count." << std::endl;
  mediapipe::Packet faceCountPacket;
  if (!internalData->faceCountPoller->Next(&faceCountPacket)) {
    std::cout << "Failed to polling face count" << std::endl;
    return;
  }
  auto faceCount = faceCountPacket.Get<int>();
  std::cout << "Polling complete. Face count : " << faceCount << std::endl;

  // Always check if there is a face you found before getting a landmark. If not, 
  // the thread can be put on standby indefinitely.
  std::cout << "Try to polling landmarks." << std::endl;
  if (faceCount != 0) {  
    mediapipe::Packet landmarksPacket;
    if (!internalData->landmarksPoller->Next(&landmarksPacket)) {
      return;
    }

    auto& landmarksVector = landmarksPacket.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    int landmarksSize = landmarksVector.size() * 3 * LANDMARK_COUNT; // x, y, z. Ignore visibility.
    
    std::vector<float> outputVector;
    outputVector.resize(landmarksSize);

    for (int i = 0 ; i < landmarksVector.size() ; i++) {
      auto& landmarks = landmarksVector[i];
      for (int j = 0 ; j < LANDMARK_COUNT ; j++) {
        auto landmark = landmarks.landmark(j);
        outputVector.push_back(landmark.x());
        outputVector.push_back(landmark.y());
        outputVector.push_back(landmark.z());
      }
    }
    
    // convert to float array.
    *outLandmarksData = outputVector.data();
    *outFaceCount = faceCount;
  }
  std::cout << "Polling landmarks complete." << std::endl;
}

void shutDownGraph(FaceMeshDataStructure** data) {
  auto internalData = (InternalData*)((*data)->internal);
  auto status = internalData->graph.CloseInputStream(kInputVideoStream);
  if (status.ok()) { 
    status = internalData->graph.WaitUntilDone();
  }
  delete internalData;
  delete (*data);
  *data = nullptr;
}
