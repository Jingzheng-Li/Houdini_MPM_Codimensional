# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build

# Include any dependencies generated for this target.
include libWetCloth/App/CMakeFiles/WetClothApp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.make

# Include the progress variables for this target.
include libWetCloth/App/CMakeFiles/WetClothApp.dir/progress.make

# Include the compile flags for this target's objects.
include libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make

libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o: ../libWetCloth/App/Camera.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o -MF CMakeFiles/WetClothApp.dir/Camera.cpp.o.d -o CMakeFiles/WetClothApp.dir/Camera.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Camera.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/Camera.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Camera.cpp > CMakeFiles/WetClothApp.dir/Camera.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/Camera.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Camera.cpp -o CMakeFiles/WetClothApp.dir/Camera.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o: ../libWetCloth/App/Main.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o -MF CMakeFiles/WetClothApp.dir/Main.cpp.o.d -o CMakeFiles/WetClothApp.dir/Main.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Main.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/Main.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Main.cpp > CMakeFiles/WetClothApp.dir/Main.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/Main.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/Main.cpp -o CMakeFiles/WetClothApp.dir/Main.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o: ../libWetCloth/App/ParticleSimulation.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o -MF CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o.d -o CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/ParticleSimulation.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/ParticleSimulation.cpp > CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/ParticleSimulation.cpp -o CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o: ../libWetCloth/App/RenderingUtilities.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o -MF CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o.d -o CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/RenderingUtilities.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/RenderingUtilities.cpp > CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/RenderingUtilities.cpp -o CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o: ../libWetCloth/App/TwoDSceneRenderer.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o -MF CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o.d -o CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneRenderer.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneRenderer.cpp > CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneRenderer.cpp -o CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o: ../libWetCloth/App/TwoDSceneSerializer.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o -MF CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o.d -o CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneSerializer.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneSerializer.cpp > CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneSerializer.cpp -o CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o: ../libWetCloth/App/TwoDSceneXMLParser.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o -MF CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o.d -o CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneXMLParser.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneXMLParser.cpp > CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDSceneXMLParser.cpp -o CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o: ../libWetCloth/App/TwoDimensionalDisplayController.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o -MF CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o.d -o CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDimensionalDisplayController.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDimensionalDisplayController.cpp > CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/TwoDimensionalDisplayController.cpp -o CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.s

libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/flags.make
libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o: ../libWetCloth/App/YImage.cpp
libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o: libWetCloth/App/CMakeFiles/WetClothApp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o -MF CMakeFiles/WetClothApp.dir/YImage.cpp.o.d -o CMakeFiles/WetClothApp.dir/YImage.cpp.o -c /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/YImage.cpp

libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/WetClothApp.dir/YImage.cpp.i"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/YImage.cpp > CMakeFiles/WetClothApp.dir/YImage.cpp.i

libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/WetClothApp.dir/YImage.cpp.s"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App/YImage.cpp -o CMakeFiles/WetClothApp.dir/YImage.cpp.s

# Object files for target WetClothApp
WetClothApp_OBJECTS = \
"CMakeFiles/WetClothApp.dir/Camera.cpp.o" \
"CMakeFiles/WetClothApp.dir/Main.cpp.o" \
"CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o" \
"CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o" \
"CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o" \
"CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o" \
"CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o" \
"CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o" \
"CMakeFiles/WetClothApp.dir/YImage.cpp.o"

# External object files for target WetClothApp
WetClothApp_EXTERNAL_OBJECTS =

libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/Camera.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/Main.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/ParticleSimulation.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/RenderingUtilities.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneRenderer.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneSerializer.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDSceneXMLParser.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/TwoDimensionalDisplayController.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/YImage.cpp.o
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/build.make
libWetCloth/App/WetClothApp: libWetCloth/Core/libWetCloth.a
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libGL.so
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libGLU.so
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libtbb.so
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libglut.so
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libpng.so
libWetCloth/App/WetClothApp: /usr/lib/x86_64-linux-gnu/libz.so
libWetCloth/App/WetClothApp: /usr/local/lib/libAntTweakBar.so
libWetCloth/App/WetClothApp: libWetCloth/App/CMakeFiles/WetClothApp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable WetClothApp"
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/WetClothApp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libWetCloth/App/CMakeFiles/WetClothApp.dir/build: libWetCloth/App/WetClothApp
.PHONY : libWetCloth/App/CMakeFiles/WetClothApp.dir/build

libWetCloth/App/CMakeFiles/WetClothApp.dir/clean:
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App && $(CMAKE_COMMAND) -P CMakeFiles/WetClothApp.dir/cmake_clean.cmake
.PHONY : libWetCloth/App/CMakeFiles/WetClothApp.dir/clean

libWetCloth/App/CMakeFiles/WetClothApp.dir/depend:
	cd /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/libWetCloth/App /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App /home/jingzheng/Public/Study/CppMine/Houdini_Libwetcloth/build/libWetCloth/App/CMakeFiles/WetClothApp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libWetCloth/App/CMakeFiles/WetClothApp.dir/depend

