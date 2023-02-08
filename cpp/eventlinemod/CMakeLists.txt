add_library(EventLinemod SHARED EventLinemod.cpp)

target_include_directories(EventLinemod INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
