enable_testing()
include_directories(../include)

function(TEST name args)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} sst opencv_imgproc opencv_imgcodecs opencv_core opencv_video opencv_videoio opencv_highgui)
    target_compile_options(${name} PRIVATE -g -O0)
    target_link_options(${name} PRIVATE)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 11)

    set(args_ls)
    set(index 1)
    while (index LESS ${ARGC})
        list(APPEND args_ls ${ARGV${index}})
        math(EXPR index "${index}+1")
    endwhile ()

    add_test(NAME ${name} WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${name} ${args_ls})
endfunction()