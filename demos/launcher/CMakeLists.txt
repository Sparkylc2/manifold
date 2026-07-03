add_executable(manifold main.cpp ../../src/app/raygui_impl.cpp)
target_link_libraries(manifold PRIVATE manifold_solver manifold_control manifold_pde manifold_renderer raygui)
target_include_directories(manifold PRIVATE ${CMAKE_SOURCE_DIR}/include)

# copy assets next to executable
add_custom_command(TARGET manifold POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:manifold>/assets
    COMMENT "Copying assets to build directory"
)
