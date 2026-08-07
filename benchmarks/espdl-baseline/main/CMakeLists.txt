# The vendored ESP-DL asm lives in the s3 crate and is compiled in directly
# (no local copy). assembler-with-cpp is required because the files use the
# preprocessor and self-#include dl_tie728_s8.S guarded by .ifndef.
set(ASM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../hematite-s3/src/asm")

idf_component_register(SRCS "main.c"
                       "${ASM_DIR}/dl_tie728_s8.S"
                       "${ASM_DIR}/dl_tie728_s8_conv2d.S"
                       "${ASM_DIR}/dl_tie728_s8_relu.S"
                       "${ASM_DIR}/dl_tie728_s8_max_pool2d.S"
                       "${ASM_DIR}/dl_tie728_s8_avg_pool2d.S"
                       "${ASM_DIR}/dl_tie728_s8_add.S"
                       "${ASM_DIR}/dl_tie728_s8_mul.S"
                       "${ASM_DIR}/dl_tie728_s8_sub.S"
                       INCLUDE_DIRS ".")

# Xtensa target builds default to windowed ABI (call8) which is the convention
# the asm uses. Force -mtext-section-literals like ESP-DL does.
set_source_files_properties("${ASM_DIR}/dl_tie728_s8.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_conv2d.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_relu.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_max_pool2d.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_avg_pool2d.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_add.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_mul.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
set_source_files_properties("${ASM_DIR}/dl_tie728_s8_sub.S" PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
