cmake_minimum_required(VERSION 3.16)
project(step_render CXX)

add_executable(step_render render.cpp)

target_include_directories(step_render PRIVATE
  /usr/include/opencascade
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(step_render PRIVATE
  TKernel TKMath TKBRep TKGeomBase TKGeomAlgo
  TKTopAlgo TKShHealing TKG2d TKG3d
  TKV3d TKOpenGl TKService
  TKSTEP TKSTEPBase TKSTEPAttr TKSTEP209 TKXSBase
  TKMesh TKHLR TKPrim
  X11 Xext GL
)