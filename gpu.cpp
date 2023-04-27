/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/gpu.hpp>
#include <algorithm>

static inline void gpu_clear(GPUMemory &mem, const ClearCommand &cmd);

static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    uint32_t draw_id
);

template<typename type, bool use_indexer>
static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    const uint32_t draw_id
);

static inline size_t extract_attributes(
    size_t *attrib_indexes,
    size_t *attrib_sizes,
    size_t *attrib_strides,
    const char **attrib_arrays,
    const VertexAttrib *attributes,
    const uint32_t max_attrib,
    const Buffer *buffers
);

//! [gpu_execute]
void gpu_execute(GPUMemory &mem, CommandBuffer &cb) {
    uint32_t draw_id = UINT32_MAX;

    for (size_t i = 0; i < cb.nofCommands; ++i) {
        switch (cb.commands[i].type) {
        case CommandType::CLEAR:
            gpu_clear(mem, cb.commands[i].data.clearCommand);
            break;
        case CommandType::DRAW:
            gpu_draw(mem, cb.commands[i].data.drawCommand, ++draw_id);
            break;
        default:
            break;
        }
    }
}
//! [gpu_execute]

static inline void gpu_clear(GPUMemory &mem, const ClearCommand &cmd) {
    if (cmd.clearColor) {
        const uint8_t comp[] = {
            static_cast<uint8_t>(cmd.color.r * 255),
            static_cast<uint8_t>(cmd.color.g * 255),
            static_cast<uint8_t>(cmd.color.b * 255),
            static_cast<uint8_t>(cmd.color.a * 255),
        };

        std::fill_n(
            reinterpret_cast<uint32_t *>(mem.framebuffer.color),
            mem.framebuffer.height * mem.framebuffer.width,
            *reinterpret_cast<const uint32_t *>(comp)
        );
    }
    if (cmd.clearDepth) {
        std::fill_n(
            mem.framebuffer.depth,
            mem.framebuffer.height * mem.framebuffer.width,
            cmd.depth
        );
    }
}

// wrapper for the template funciton
static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    const uint32_t draw_id
) {
    if (cmd.vao.indexBufferID < 0) {
        gpu_draw<void, false>(mem, cmd, draw_id);
        return;
    }

    switch (cmd.vao.indexType) {
    case IndexType::UINT8:
        gpu_draw<uint8_t, true>(mem, cmd, draw_id);
        return;
    case IndexType::UINT16:
        gpu_draw<uint16_t, true>(mem, cmd, draw_id);
        return;
    case IndexType::UINT32:
        gpu_draw<uint32_t, true>(mem, cmd, draw_id);
        return;
    }
}

// use template for the draw function to avoid duplicate code but don't
// sacriface performance
template<typename type, bool use_indexer>
static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    const uint32_t draw_id
) {
    const Program &prog = mem.programs[cmd.programID];

    InVertex in_vertex{
        .gl_DrawID = draw_id
    };

    ShaderInterface si{
        .uniforms = mem.uniforms,
        .textures = mem.textures,
    };

    // this is left out when not using indexer
    const type *indexer;
    if constexpr(use_indexer) {
        indexer = reinterpret_cast<const type *>(
            reinterpret_cast<const char *>(
                mem.buffers[cmd.vao.indexBufferID].data
            ) + cmd.vao.indexOffset
        );
    }

    // attributes
    size_t at_ind[maxAttributes];
    size_t at_siz[maxAttributes];
    size_t at_str[maxAttributes];
    const char *at_arr[maxAttributes];

    size_t at_cnt = extract_attributes(
        at_ind, at_siz, at_str, at_arr,
        cmd.vao.vertexAttrib, maxAttributes, mem.buffers
    );

    for (size_t i = 0; i < cmd.nofVertices; ++i) {
        OutVertex out_vertex;

        // set the index based on whether to use indexer
        if constexpr(use_indexer) {
            in_vertex.gl_VertexID = indexer[i];
        } else {
            in_vertex.gl_VertexID = i;
        }

        // set the attributes
        for (size_t j = 0; j < at_cnt; ++j) {
            std::copy_n(
                at_arr[j] + (in_vertex.gl_VertexID * at_str[j]),
                at_siz[j],
                reinterpret_cast<char *>(&in_vertex.attributes[at_ind[j]])
            );
        }

        prog.vertexShader(out_vertex, in_vertex, si);
    }
}

static inline size_t extract_attributes(
    size_t *attrib_indexes,
    size_t *attrib_sizes,
    size_t *attrib_strides,
    const char **attrib_arrays,
    const VertexAttrib *attributes,
    const uint32_t max_attrib,
    const Buffer *buffers
) {
    size_t attrib_count = 0;

    for (size_t i = 0; i < max_attrib; ++i) {
        // filter out unused attributes
        if (attributes[i].type == AttributeType::EMPTY)
            continue;

        attrib_indexes[attrib_count] = i;
        attrib_sizes[attrib_count] =
            static_cast<size_t>(attributes[i].type) % 8 * 4;

        attrib_strides[attrib_count] = attributes[i].stride;
        attrib_arrays[attrib_count] = reinterpret_cast<const char *>(
            buffers[attributes[i].bufferID].data
        ) + attributes[i].offset;
        ++attrib_count;
    }

    return attrib_count;
}

/**
 * @brief This function reads color from texture.
 *
 * @param texture texture
 * @param uv uv coordinates
 *
 * @return color 4 floats
 */
glm::vec4 read_texture(Texture const &texture, glm::vec2 uv) {
    if(!texture.data)return glm::vec4(0.f);
    auto uv1 = glm::fract(uv);
    auto uv2 = uv1*glm::vec2(texture.width-1,texture.height-1)+0.5f;
    auto pix = glm::uvec2(uv2);
    //auto t   = glm::fract(uv2);
    glm::vec4 color = glm::vec4(0.f,0.f,0.f,1.f);
    for(uint32_t c=0;c<texture.channels;++c)
        color[c] = texture.data[(pix.y*texture.width+pix.x)*texture.channels+c]/255.f;
    return color;
}

