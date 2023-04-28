/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/gpu.hpp>
#include <algorithm>

struct ExtAttrib {
    inline ExtAttrib(const VertexAttrib *attributes, const Buffer *buffers);
    inline void set_attrib(size_t index, Attribute *out_attribs) const;
    // the actual indexes
    size_t ind[maxAttributes];
    // sizes of the data
    size_t siz[maxAttributes];
    // strides
    size_t str[maxAttributes];
    // the coresponding buffers
    const char *arr[maxAttributes];
    // number of attributes
    size_t cnt;
};

#define DRAW_INDEXER 0x1
#define DRAW_CULLING 0x2

// clears the color/depth buffer based on the command
static inline void gpu_clear(GPUMemory &mem, const ClearCommand &cmd);

// draws triangles based on the command (wrapper for the other gpu_draw)
static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    uint32_t draw_id
);

/**
 * @brief template fro drawing
 *
 * @param index type when using indexer
 * @param flags used to enable some features (indexer culling)
 */
template<typename type, int flags>
static void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    const uint32_t draw_id
);

// determines whether the triangle is not facing the camera
static inline bool is_backface(OutVertex *triangle);

// rasterizes the given triangle (ndc coordinates) on the frame
static inline void rasterize(Frame &frame, OutVertex *triangle);

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
        cmd.backfaceCulling
            ? gpu_draw<void, DRAW_CULLING>(mem, cmd, draw_id)
            : gpu_draw<void, 0>(mem, cmd, draw_id);
        return;
    }

    switch (cmd.vao.indexType) {
    case IndexType::UINT8:
        cmd.backfaceCulling
            ? gpu_draw<uint8_t, DRAW_INDEXER | DRAW_CULLING>(
                mem,
                cmd,
                draw_id
            ) : gpu_draw<uint8_t, DRAW_INDEXER>(mem, cmd, draw_id);
        return;
    case IndexType::UINT16:
        cmd.backfaceCulling
            ? gpu_draw<uint16_t, DRAW_INDEXER | DRAW_CULLING>(
                mem,
                cmd,
                draw_id
            ) : gpu_draw<uint16_t, DRAW_INDEXER>(mem, cmd, draw_id);
        return;
    case IndexType::UINT32:
        cmd.backfaceCulling
            ? gpu_draw<uint32_t, DRAW_INDEXER | DRAW_CULLING>(
                mem,
                cmd,
                draw_id
            ) : gpu_draw<uint32_t, DRAW_INDEXER>(mem, cmd, draw_id);
        return;
    }
}

// use template for the draw function to avoid duplicate code but don't
// sacriface performance, flags indicate the compile-time features to turn on
template<typename type, int flags>
static void gpu_draw(
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
    if constexpr(flags & DRAW_INDEXER) {
        indexer = reinterpret_cast<const type *>(
            reinterpret_cast<const char *>(
                mem.buffers[cmd.vao.indexBufferID].data
            ) + cmd.vao.indexOffset
        ) - 1;
    }

    // extract attributes
    ExtAttrib at{ cmd.vao.vertexAttrib, mem.buffers };

    // start at 2 and search the vertices backwards to avoid checks that
    // cmd.nofVertices is multiple of 3
    for (size_t i = 2; i < cmd.nofVertices; i += 3) {
        OutVertex triangle[3];

        // run the vertex shader for 3 vertices
        // j starts at 2 to ensure that the vertices are processed in order
        for (size_t j = 2; j != SIZE_MAX; --j) {
            // set the index based on whether to use indexer
            if constexpr(flags & DRAW_INDEXER)
                in_vertex.gl_VertexID = *++indexer;
            else
                in_vertex.gl_VertexID = i - j;

            // run the vertex shader
            at.set_attrib(in_vertex.gl_VertexID, in_vertex.attributes);
            prog.vertexShader(triangle[2 - j], in_vertex, si);

            // perspective division
            triangle[2 - j].gl_Position /= triangle[2 - j].gl_Position.w;
        }

        // skip backface triangles if culling is enabled
        if constexpr(flags & DRAW_CULLING) {
            if (is_backface(triangle))
                continue;
        }

        rasterize(mem.framebuffer, triangle);
    }
}

inline ExtAttrib::ExtAttrib(
    const VertexAttrib *attributes,
    const Buffer *buffers
) : cnt(0) {
    for (size_t i = 0; i < maxAttributes; ++i) {
        // filter out unused attributes
        if (attributes[i].type == AttributeType::EMPTY)
            continue;

        ind[cnt] = i;
        // type % 8 * 4
        siz[cnt] = (static_cast<size_t>(attributes[i].type) & 7) << 2;
        str[cnt] = attributes[i].stride;
        arr[cnt] = reinterpret_cast<const char *>(
            buffers[attributes[i].bufferID].data
        ) + attributes[i].offset;

        ++cnt;
    }
}

inline void ExtAttrib::set_attrib(size_t index, Attribute *out_attribs) const {
    for (size_t j = 0; j < cnt; ++j) {
        std::copy_n(
            arr[j] + (index * str[j]),
            siz[j],
            reinterpret_cast<char *>(out_attribs + ind[j])
        );
    }
}

static inline bool is_backface(OutVertex *triangle) {
    // |x0 y0 1|
    // |x1 y1 1| < 0 => clockwise => backface
    // |x2 y2 1|
    //     ||
    // x0*y1 + x1*y2 + x2*y0 < y0*x1 + y1*x2 + y2*x0 => clockwise

    auto a = // x0*y1 + x1*y2 + x2*y0
        triangle[0].gl_Position.x * triangle[1].gl_Position.y +
        triangle[1].gl_Position.x * triangle[2].gl_Position.y +
        triangle[2].gl_Position.x * triangle[0].gl_Position.y;
    auto b = // y0*x1 + y1*x2 + y2*x0
        triangle[0].gl_Position.y * triangle[1].gl_Position.x +
        triangle[1].gl_Position.y * triangle[2].gl_Position.x +
        triangle[2].gl_Position.y * triangle[0].gl_Position.x;

    return a <= b;
}

static inline void rasterize(Frame &frame, OutVertex *triangle) {
    // names for more readable code
    auto p0 = triangle[0].gl_Position;
    auto p1 = triangle[1].gl_Position;
    auto p2 = triangle[2].gl_Position;

    // viewport transform
    p0.x = (p0.x + 1) * frame.width / 2;
    p0.y = (p0.y + 1) * frame.height / 2;
    p1.x = (p1.x + 1) * frame.width / 2;
    p1.y = (p1.y + 1) * frame.height / 2;
    p2.x = (p2.x + 1) * frame.width / 2;
    p2.y = (p2.y + 1) * frame.height / 2;

    // calculate bounding box inside the screen
    glm::vec2 fbl{ // bottom left
        std::max(0.f, std::min({p0.x, p1.x, p2.x})),
        std::max(0.f, std::min({p0.y, p1.y, p2.y})),
    };
    glm::vec2 ftr{ // top right
        std::min<float>(frame.width - 1, std::max({p0.x, p1.x, p2.x})),
        std::min<float>(frame.height - 1, std::max({p0.y, p1.y, p2.y})),
    };

    // check if the triangle is on screen
    if (fbl.x >= ftr.x || fbl.y >= ftr.y)
        return;

    // make it integers
    glm::uvec2 bl = fbl;
    glm::uvec2 tr = ftr;

    // TODO: rasterization
    // rasterization using pineda

    // edge vectors
    glm::vec2 d0{ p1.x - p0.x, p1.y - p0.y };
    glm::vec2 d1{ p2.x - p1.x, p2.y - p1.y };
    glm::vec2 d2{ p0.x - p2.x, p0.y - p2.y };

    // ABGR color - yellow, temporary before baricentric coordinates
    uint32_t color = 0xFF00FFFFu;
    uint32_t *colbuf = reinterpret_cast<uint32_t *>(frame.color);

    for (glm::uint y = bl.y; y <= tr.y; ++y) {
        float e0 = (bl.x - p0.x) * d0.y - (y - p0.y) * d0.x;
        float e1 = (bl.x - p1.x) * d1.y - (y - p1.y) * d1.x;
        float e2 = (bl.x - p2.x) * d2.y - (y - p2.y) * d2.x;
        for (glm::uint x = bl.x; x <= tr.x; ++x) {
            if (e0 >= 0 && e1 >= 0 && e2 >= 0)
                colbuf[y * frame.width + x] = color;

            e0 += d0.y;
            e1 += d1.y;
            e2 += d2.y;
        }
    }
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
    if (!texture.data)
        return glm::vec4(0.f);

    auto uv1 = glm::fract(uv);
    auto uv2 = uv1 * glm::vec2(texture.width - 1, texture.height - 1) + 0.5f;
    auto pix = glm::uvec2(uv2);
    //auto t   = glm::fract(uv2);
    glm::vec4 color = glm::vec4(0.f, 0.f, 0.f, 1.f);

    for (uint32_t c = 0; c < texture.channels; ++c)
        color[c] = texture.data[
            (pix.y * texture.width + pix.x) * texture.channels + c
        ] / 255.f;

    return color;
}

