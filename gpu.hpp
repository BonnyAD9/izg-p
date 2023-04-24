/*!
 * @file
 * @brief This file contains class that represents graphic card.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#pragma once

#include <student/fwd.hpp>

/**
 * @brief function that executes work stored in command buffer on the gpu memory.
 * This function represents the functionality of GPU.
 * It can render stuff, it can clear framebuffer.
 * It can process work stored in command buffer
 *
 * @param mem gpu memory
 * @param cb command buffer - packaged of work sent to the gpu
 */
void gpu_execute(GPUMemory&mem,CommandBuffer&cb);

glm::vec4 read_texture(Texture const&texture,glm::vec2 uv);
