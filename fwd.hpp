/*!
 * @file
 * @brief This file contains forward declarations, structures and constants.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

//#define MAKE_STUDENT_RELEASE

uint32_t const maxAttributes = 4;///< maximum number of vertex/fragment attributes

/**
 * @brief This struct represent a texture
 */
//! [Texture]
struct Texture{
  uint8_t const* data     = nullptr;///< pointer to data
  uint32_t       width    = 0      ;///< width of the texture
  uint32_t       height   = 0      ;///< height of the texture
  uint32_t       channels = 3      ;///< number of channels of the texture
};
//! [Texture]

/**
 * @brief This enum represents vertex/fragment attribute type.
 */
//! [AttributeType]
enum class AttributeType{
  EMPTY =   0, ///< disabled attribute
  FLOAT =   1, ///< 1x 32-bit float
  VEC2  =   2, ///< 2x 32-bit floats
  VEC3  =   3, ///< 3x 32-bit floats
  VEC4  =   4, ///< 4x 32-bit floats
  UINT  = 8+1, ///< 1x 32-bit unsigned int
  UVEC2 = 8+2, ///< 2x 32-bit unsigned int
  UVEC3 = 8+3, ///< 3x 32-bit unsigned int
  UVEC4 = 8+4, ///< 4x 32-bit unsigned int
};
//! [AttributeType]

/**
 * @brief This union represents one vertex/fragment attribute
 */
//! [Attribute]
union Attribute{
  Attribute():v4(glm::vec4(1.f)){}
  float      v1; ///< single float
  glm::vec2  v2; ///< vector of two floats
  glm::vec3  v3; ///< vector of three floats
  glm::vec4  v4; ///< vector of four floats
  uint32_t   u1; ///< single unsigned int
  glm::uvec2 u2; ///< vector of two unsigned ints
  glm::uvec3 u3; ///< vector of three unsigned ints
  glm::uvec4 u4; ///< vector of four unsigned ints
};
//! [Attribute]

/**
 * @brief This struct represents input vertex of vertex shader.
 */
//! [InVertex]
struct InVertex{
  Attribute attributes[maxAttributes]    ; ///< vertex attributes
  uint32_t  gl_VertexID               = 0; ///< vertex id
  uint32_t  gl_DrawID                 = 0; ///< draw id
};
//! [InVertex]

/**
 * @brief This struct represents output vertex of vertex shader.
 */
//! [OutVertex]
struct OutVertex{
  Attribute attributes[maxAttributes]                     ; ///< vertex attributes
  glm::vec4 gl_Position               = glm::vec4(0,0,0,1); ///< clip space position
};
//! [OutVertex]

/**
 * @brief This struct represents input fragment.
 */
//! [InFragment]
struct InFragment{
  Attribute attributes[maxAttributes]               ; ///< fragment attributes
  glm::vec4 gl_FragCoord              = glm::vec4(1); ///< fragment coordinates
};
//! [InFragment]

/**
 * @brief This struct represents output fragment.
 */
//! [OutFragment]
struct OutFragment{
  glm::vec4 gl_FragColor = glm::vec4(0.f); ///< fragment color
};
//! [OutFragment]

/**
 * @brief This union represents one uniform variable.
 */
//! [Uniform]
union Uniform{
  Uniform(){}
  float      v1; ///< single float
  glm::vec2  v2; ///< two    floats
  glm::vec3  v3; ///< three  floats
  glm::vec4  v4; ///< four   floats
  uint32_t   u1; ///< single 32bit unsigned int
  glm::uvec2 u2; ///< two    32bit unsigned ints
  glm::uvec3 u3; ///< three  32bit unsigned ints
  glm::uvec4 u4; ///< four   32bit unsigned ints
  int32_t    i1; ///< single 32bit int
  glm::ivec2 i2; ///< two    32bit ints
  glm::ivec3 i3; ///< three  32bit ints
  glm::ivec4 i4; ///< four   32bit ints
  glm::mat4  m4 = glm::mat4(1.f); ///< 4x4 float matrix
};
//! [Uniform]

/**
 * @brief This enum represents index type
 */
//! [IndexType]
enum class IndexType{
  UINT8  = 1, ///< uin8_t type
  UINT16 = 2, ///< uin16_t type
  UINT32 = 4, ///< uint32_t type
};
//! [IndexType]


/**
 * @brief This enum represents constant shader interface common for all shaders.
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param uniforms uniform variables
 */
//! [ShaderInterface]
struct ShaderInterface{
  Uniform const*uniforms = nullptr; ///< uniform variables
  Texture const*textures = nullptr; ///< textures
};
//! [ShaderInterface]

/**
 * @brief Function type for vertex shader
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param uniforms uniform variables
 */
//! [VertexShader]
using VertexShader = void(*)(
    OutVertex            &outVertex,
    InVertex        const&inVertex ,
    ShaderInterface const&si       );
//! [VertexShader]

/**
 * @brief Function type for fragment shader
 *
 * @param outFragment output fragment
 * @param inFragment input fragment 
 * @param uniforms uniform variables
 */
//! [FragmentShader]
using FragmentShader = void(*)(
    OutFragment          &outFragment,
    InFragment      const&inFragment ,
    ShaderInterface const&si         );
//! [FragmentShader]

/**
 * @brief This struct describes location of one vertex attribute.
 */
//! [VertexAttrib]
struct VertexAttrib{
  int32_t       bufferID = -1                  ;///< buffer id
  uint64_t      stride   = 0                   ;///< stride in bytes
  uint64_t      offset   = 0                   ;///< offset in bytes
  AttributeType type     = AttributeType::EMPTY;///< type of attribute
};
//! [VertexAttrib]

/**
 * @brief This structure represents setting for vertex pulller (vertex assembly) unit.
 * VertexArrays holds setting for reading vertices from buffers.
 */
//! [VertexArray]
struct VertexArray{
  VertexAttrib vertexAttrib[maxAttributes];       ///< settings for vertex attributes
  int32_t      indexBufferID = -1;                ///< id of index buffer
  uint64_t     indexOffset   = 0 ;                ///< offset of indices
  IndexType    indexType     = IndexType::UINT32; ///< type of indices
};
//! [VertexArray]

/**
 * @brief This structu represents a program.
 * Vertex Shader is executed on every InVertex.
 * Fragment Shader is executed on every rasterized InFragment.
 */
//! [Program]
struct Program{
  VertexShader   vertexShader   = nullptr; ///< vertex shader
  FragmentShader fragmentShader = nullptr; ///< fragment shader
  AttributeType  vs2fs[maxAttributes] = {AttributeType::EMPTY}; ///< which attributes are interpolated from vertex shader to fragment shader
};
//! [Program]

/**
 * @brief This structure represents a frame.
 * Frame (or framebuffer) is used as output of rendering.
 */
//! [Frame]
struct Frame{
  uint8_t* color    = nullptr; ///< color buffer (1 byte per channel)
  float  * depth    = nullptr; ///< depth buffer
  uint32_t channels = 4      ; ///< number of color channels
  uint32_t width    = 0      ; ///< width of frame
  uint32_t height   = 0      ; ///< height of frame
};
//! [Frame]


/**
 * @brief This structure represents a buffer on GPU
 * Buffer is a linear memory.
 */
//! [Buffer]
struct Buffer{
  void const* data = nullptr; ///< pointer to data
  uint64_t    size = 0      ; ///< size of data in bytes
};
//! [Buffer]

/**
 * @brief This structure represents memory on GPU
 */
//! [GPUMemory]
struct GPUMemory{
  uint32_t const static maxUniforms = 10000; ///< maximal number of uniforms
  uint32_t const static maxTextures = 1000 ; ///< maximal number of textures
  uint32_t const static maxBuffers  = 100  ; ///< maximal number of buffers
  uint32_t const static maxPrograms = 100  ; ///< maximal number of programs
  Buffer  buffers [maxBuffers ]; ///< array of all buffers
  Texture textures[maxTextures]; ///< array of all textures
  Uniform uniforms[maxUniforms]; ///< array of all uniform variables
  Program programs[maxPrograms]; ///< array of all programs
  Frame   framebuffer;           ///< framebuffer - output of rendering
};
//! [GPUMemory]




/**
 * @brief This structure represents clear command.
 * Clear command stores data which are used by clearing operation on the GPU.
 */
//! [ClearCommand]
struct ClearCommand{
  glm::vec4   color      = glm::vec4(0); ///< color buffer will be cleared by this value
  float       depth      = 1e10        ; ///< depth buffer will be cleared by this value
  bool        clearColor = true        ; ///< is color cleaning enabled?
  bool        clearDepth = true        ; ///< is depth cleaning enabled?
};
//! [ClearCommand]

/**
 * @brief This structure represents draw command.
 * Draw command issues draw operation on the GPU.
 */
//! [DrawCommand]
struct DrawCommand{
  int32_t     programID       = -1   ; ///< selected shader program - id
  uint32_t    nofVertices     = 0    ; ///< number of vertices to draw
  bool        backfaceCulling = false; ///< is culling of backfacing triangles enabled?
  VertexArray vao                    ; ///< active vertex array (input/ triangles)
};
//! [DrawCommand]

/**
 * @brief This enum represents type of command.
 */
//! [CommandType]
enum class CommandType{
  EMPTY, ///< empty command
  CLEAR, ///< clear command
  DRAW , ///< draw command
};
//! [CommandType]

/**
 * @brief This union represents command data.
 */
//! [CommandData]
union CommandData{
  CommandData():drawCommand(){}///< constructor
  ClearCommand clearCommand;   ///< clear command data
  DrawCommand  drawCommand ;   ///< draw command data
};
//! [CommandData]

/**
 * @brief This struct represents a command.
 */
//! [Command]
struct Command{
  CommandData data                     ;///< data of the command
  CommandType type = CommandType::EMPTY;///< type of the command
};
//! [Command]

/**
 * @brief This struct represents a command buffer.
 * Command buffer is used for CPU -> GPU communication.
 * Jobs are sent to GPU using this structure.
 */
//! [CommandBuffer]
struct CommandBuffer{
  uint32_t static const maxCommands           = 10000; ///< maximal number of commands
  uint32_t              nofCommands           = 0    ; ///< number of used commands in command buffer
  Command               commands[maxCommands]        ; ///< array of commands
};
//! [CommandBuffer]

/**
 * @brief This function can be used to insert clear command to command buffer.
 *
 * @param cb command buffer
 * @param color color for cleaning
 * @param depth depth for cleaning
 * @param clearColor should the color buffer be cleaned?
 * @param clearDepth should the depth buffer be cleaned?
 */
inline void pushClearCommand(
    CommandBuffer      &cb                       ,
    glm::vec4     const&color      = glm::vec4(0),
    float               depth      = 10e10       ,
    bool                clearColor = true        ,
    bool                clearDepth = true        ){
  auto&cmd=cb.commands[cb.nofCommands];
  cmd.type = CommandType::CLEAR;
  auto&c = cmd.data.clearCommand;
  c.color      = color;
  c.depth      = depth     ;
  c.clearColor = clearColor;
  c.clearDepth = clearDepth;
  cb.nofCommands++;
}

/**
 * @brief This function can be used to insert daw command into command buffer.
 *
 * @param cb command buffer
 * @param nofVertices number of vertices that should be rendered
 * @param prg index of program that should be used for rendering
 * @param vao vertex array
 * @param backfaceCulling should the backface culling be enabled?
 */
inline void pushDrawCommand(
    CommandBuffer      &cb                     ,
    uint32_t            nofVertices            , 
    int32_t             prg             = 0    ,
    VertexArray   const&vao             = {}   ,
    bool                backfaceCulling = false){
  auto&cmd=cb.commands[cb.nofCommands];
  cmd.type = CommandType::DRAW;
  auto&c = cmd.data.drawCommand;
  c.backfaceCulling = backfaceCulling;
  c.nofVertices     = nofVertices    ;
  c.programID       = prg            ;
  c.vao             = vao            ;
  cb.nofCommands++;
}



/**
 * @brief This struct represents a mesh
 */
//! [Mesh]
struct Mesh{
  int32_t      indexBufferID  = -1               ;///< index of buffer used for indices
  size_t       indexOffset    = 0                ;///< offset into index buffer
  IndexType    indexType      = IndexType::UINT32;///< type of indices
  VertexAttrib position                          ;///< position vertex attribute
  VertexAttrib normal                            ;///< normal vertex attribute
  VertexAttrib texCoord                          ;///< tex. coord vertex attribute
  uint32_t     nofIndices     = 0                ;///< nofIndices or nofVertices (if there is no indexing)
  glm::vec4    diffuseColor   = glm::vec4(1.f)   ;///< default diffuseColor (if there is no texture)
  int          diffuseTexture = -1               ;///< diffuse texture or -1 (no texture)
  bool         doubleSided    = false            ;///< double sided material
};
//! [Mesh]

/**
 * @brief This structure represents node in tree structure of model
 */
//! [Node]
struct Node{
  glm::mat4        modelMatrix = glm::mat4(1.f);///< model transformation matrix
  int32_t          mesh        = -1;            ///< id of mesh or -1 if no mesh
  std::vector<Node>children;                    ///< list of children nodes
};
//! [Node]

/**
 * @brief This struct represent model
 */
//! [Model]
struct Model{
  std::vector<Mesh   >meshes  ;///< list of all meshes in model
  std::vector<Node   >roots   ;///< list of roots of node trees
  std::vector<Texture>textures;///< list of all textures in model
  std::vector<Buffer> buffers ;///< list of all buffers in model
};
//! [Model]
