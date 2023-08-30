# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zzu/scene_seal/align-cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zzu/scene_seal/align-cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/sealnet_align.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sealnet_align.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sealnet_align.dir/flags.make

CMakeFiles/sealnet_align.dir/main.cu.o: CMakeFiles/sealnet_align.dir/flags.make
CMakeFiles/sealnet_align.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzu/scene_seal/align-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sealnet_align.dir/main.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/zzu/scene_seal/align-cuda/main.cu -o CMakeFiles/sealnet_align.dir/main.cu.o

CMakeFiles/sealnet_align.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sealnet_align.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sealnet_align.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sealnet_align.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/sealnet_align.dir/main.cu.o.requires:

.PHONY : CMakeFiles/sealnet_align.dir/main.cu.o.requires

CMakeFiles/sealnet_align.dir/main.cu.o.provides: CMakeFiles/sealnet_align.dir/main.cu.o.requires
	$(MAKE) -f CMakeFiles/sealnet_align.dir/build.make CMakeFiles/sealnet_align.dir/main.cu.o.provides.build
.PHONY : CMakeFiles/sealnet_align.dir/main.cu.o.provides

CMakeFiles/sealnet_align.dir/main.cu.o.provides.build: CMakeFiles/sealnet_align.dir/main.cu.o


# Object files for target sealnet_align
sealnet_align_OBJECTS = \
"CMakeFiles/sealnet_align.dir/main.cu.o"

# External object files for target sealnet_align
sealnet_align_EXTERNAL_OBJECTS =

CMakeFiles/sealnet_align.dir/cmake_device_link.o: CMakeFiles/sealnet_align.dir/main.cu.o
CMakeFiles/sealnet_align.dir/cmake_device_link.o: CMakeFiles/sealnet_align.dir/build.make
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_gapi.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_highgui.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_ml.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_objdetect.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_photo.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_stitching.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_video.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_videoio.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /home/zzu/anaconda3/lib/libpython3.9.so
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcudart_static.a
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.so
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgcodecs.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_dnn.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_calib3d.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_features2d.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_flann.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_imgproc.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: /usr/local/lib/libopencv_core.so.4.8.0
CMakeFiles/sealnet_align.dir/cmake_device_link.o: CMakeFiles/sealnet_align.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzu/scene_seal/align-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/sealnet_align.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sealnet_align.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sealnet_align.dir/build: CMakeFiles/sealnet_align.dir/cmake_device_link.o

.PHONY : CMakeFiles/sealnet_align.dir/build

# Object files for target sealnet_align
sealnet_align_OBJECTS = \
"CMakeFiles/sealnet_align.dir/main.cu.o"

# External object files for target sealnet_align
sealnet_align_EXTERNAL_OBJECTS =

sealnet_align.so: CMakeFiles/sealnet_align.dir/main.cu.o
sealnet_align.so: CMakeFiles/sealnet_align.dir/build.make
sealnet_align.so: /usr/local/lib/libopencv_gapi.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_highgui.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_ml.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_objdetect.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_photo.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_stitching.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_video.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_videoio.so.4.8.0
sealnet_align.so: /home/zzu/anaconda3/lib/libpython3.9.so
sealnet_align.so: /usr/local/cuda/lib64/libcudart_static.a
sealnet_align.so: /usr/lib/x86_64-linux-gnu/librt.so
sealnet_align.so: /usr/local/lib/libopencv_imgcodecs.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_dnn.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_calib3d.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_features2d.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_flann.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_imgproc.so.4.8.0
sealnet_align.so: /usr/local/lib/libopencv_core.so.4.8.0
sealnet_align.so: CMakeFiles/sealnet_align.dir/cmake_device_link.o
sealnet_align.so: CMakeFiles/sealnet_align.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzu/scene_seal/align-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA shared module sealnet_align.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sealnet_align.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sealnet_align.dir/build: sealnet_align.so

.PHONY : CMakeFiles/sealnet_align.dir/build

CMakeFiles/sealnet_align.dir/requires: CMakeFiles/sealnet_align.dir/main.cu.o.requires

.PHONY : CMakeFiles/sealnet_align.dir/requires

CMakeFiles/sealnet_align.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sealnet_align.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sealnet_align.dir/clean

CMakeFiles/sealnet_align.dir/depend:
	cd /home/zzu/scene_seal/align-cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzu/scene_seal/align-cuda /home/zzu/scene_seal/align-cuda /home/zzu/scene_seal/align-cuda/build /home/zzu/scene_seal/align-cuda/build /home/zzu/scene_seal/align-cuda/build/CMakeFiles/sealnet_align.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sealnet_align.dir/depend
