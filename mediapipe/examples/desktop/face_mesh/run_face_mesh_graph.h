#ifndef RUN_FACE_MESH_GRAPH_H_
#define RUN_FACE_MESH_GRAPH_H_

#define LANDMARK_COUNT 468

// Library for run DNN base face mesh landmarks.
// Limitation. (According to what is known in the media pipe example.)
//  - Only support RGB format.
//  - GPU support on mobile platform. ()
//  - The ImageFrame carrier class supports more types of formats, but the model seems 
//    to only support RGB. And the ImageFrame class does not support format conversion 
//    by default.
//  - The paths of face_detection_front.tflite and face_detection_front.tflite must 
//    be specified correctly. You can modify paths in face_landmark_{platform}.pbtxt
//    and face_detection_front_{platform}.pbtxt. And it must be specified at library build time.



// Structure for save data and interact with library.
typedef struct FaceMeshDataStructure_ {
    void* internal;
} FaceMeshDataStructure;

// For common usage. See above about supported pixel format.
enum PixelFormat {
    SRGB = 0,
    SRGBA,
    SBGRA
};

// Copy : Internally use memcpy. Slow but safe.
// SHARE : Sharing imege data ptr. Not deallocated when inference complete.
// TRANSFER : Model take ownership of image data, when inference complete 
//           data will be release.

enum DataTransferMethod {
    COPY = 0,
    SHARE,
    TRANSFER
};

void initGraph(FaceMeshDataStructure** data);
void RunGraph(FaceMeshDataStructure* data, const PixelFormat pixelformat, 
    const int width, const int height, const bool useCopy,
    const unsigned char* imageData, int* outFaceCount, float** outLandmarksData);
void shutDownGraph(FaceMeshDataStructure** data);

#endif
