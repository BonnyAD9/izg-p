/*!
 * @file
 * @brief This file contains function declarations for model rendering
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#pragma once

#include <student/fwd.hpp>

//void drawModel(Frame&frame,Model const&model,glm::mat4 const&proj,glm::mat4 const&view,glm::vec3 const&light,glm::vec3 const&camera);

void prepareModel(GPUMemory&mem,CommandBuffer&commandBuffer,Model const&model);

void drawModel_vertexShader(OutVertex&outVertex,InVertex const&inVertex,ShaderInterface const&si);

void drawModel_fragmentShader(OutFragment&outFragment,InFragment const&inFragment,ShaderInterface const&si);
