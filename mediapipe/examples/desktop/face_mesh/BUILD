# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "face_mesh_tflite",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_calculators",
    ],
)

cc_binary(
    name = "face_mesh_cpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
    ],
)

# Linux only
cc_binary(
    name = "face_mesh_gpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
    ],
)

# Landmark test.
cc_binary(
    name = "face_mesh_cpu_landmark",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_test",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",      
        "//mediapipe/examples/desktop/face_mesh/calculators:detected_face_count_calculator"
    ],
)

# For face mesh library build.
# CPU / Face landmark only.
cc_binary(
    name = "face_mesh_cpu.so",
    srcs = ["run_face_mesh_graph.cc", "run_face_mesh_graph.h",],
    visibility = ["//visibility:public"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        # Run graph.
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        # Graph contorl node.
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        # Face landmark calculation module.
        # This include calculator, graph, tensorflow.
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/examples/desktop/face_mesh/calculators:detected_face_count_calculator"
    ],
)