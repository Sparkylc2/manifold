add_executable(manifold main.cpp ../../src/app/raygui_impl.cpp)
target_link_libraries(manifold PRIVATE manifold_demos)
target_include_directories(manifold PRIVATE ${CMAKE_SOURCE_DIR}/include)

# copy assets next to executable
add_custom_command(TARGET manifold POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:manifold>/assets
    COMMENT "Copying assets to build directory"
)

# POST_BUILD only fires when the exe relinks, so editing a shader alone left a
# stale copy running and the change looked like it had no effect. Shaders are a
# few KB, so re-copy them every build; the 57 MB of fonts and art stay on the
# relink hook.
add_custom_target(manifold_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets/shaders $<TARGET_FILE_DIR:manifold>/assets/shaders
    COMMENT "Refreshing shaders"
)
add_dependencies(manifold_shaders manifold)
