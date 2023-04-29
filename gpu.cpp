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

struct Triangle {
    inline Triangle(glm::vec4 a, glm::vec4 b, glm::vec4 c);
    // determines whether the triangle is backface to the camera or not
    inline bool is_backface() const;
    // gets the area of the triangle
    // the template parameter is there to optimize
    template<bool backface>
    inline float get_area();

    // points of the triangle
    glm::vec4 a;
    glm::vec4 b;
    glm::vec4 c;

    // vectors of the triangle sides (in xy 2D)
    // used in many computation -> they are precomputed here
    // ab = vector from a to b
    glm::vec2 ab;
    glm::vec2 bc;
    glm::vec2 ca;

    // area that can be procomputed using the get_area method
    float area;
};

struct FragmentContext {
    inline FragmentContext(
        const Triangle &t,
        const OutVertex *vert,
        Frame &frame,
        const Program &prog,
        const ShaderInterface &si,
        bool &failed
    );
    inline void eval_at(const float x, const float y);
    inline void add_x();
    inline void sub_x();
    inline void add_y();
    inline void sub_y();
    inline void draw();
    template<bool backface>
    inline bool should_draw() const;

    // tirangle to draw
    const Triangle &t;
    // the results of vertex shader
    const OutVertex *vert;
    // the shader program
    const Program &prog;
    // the constants for shader
    const ShaderInterface &si;
    // the frame buffer
    Frame &frame;
    // the current pixel
    glm::vec2 pt;
    // the color buffer
    uint32_t *color;
    // values of triangle side equations of pt
    float abv;
    float bcv;
    float cav;
    // bottom left bounding box coordinate
    glm::ivec2 bl;
    // top right bounding box coordinate
    glm::ivec2 tr;

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
static inline void gpu_draw(
    GPUMemory &mem,
    const DrawCommand &cmd,
    const uint32_t draw_id
);

// rasterizes the given triangle (ndc coordinates) on the frame
template<bool backface>
static inline void rasterize(
    Frame &frame,
    OutVertex *triangle,
    const Program &prog,
    const ShaderInterface &si
);

static inline uint32_t to_rgba(glm::vec4 color);

// get the area of a triangle
static inline float triangle_area(glm::vec2 a, glm::vec2 b, glm::vec2 c);

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
        std::fill_n(
            reinterpret_cast<uint32_t *>(mem.framebuffer.color),
            mem.framebuffer.height * mem.framebuffer.width,
            to_rgba(cmd.color)
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
            triangle[2 - j].gl_Position.x /= triangle[2 - j].gl_Position.w;
            triangle[2 - j].gl_Position.y /= triangle[2 - j].gl_Position.w;
            triangle[2 - j].gl_Position.z /= triangle[2 - j].gl_Position.w;
        }

        Triangle t{
            triangle[0].gl_Position,
            triangle[1].gl_Position,
            triangle[2].gl_Position
        };

        if (t.is_backface()) {
            // skip backface triangles if culling is enabled
            if constexpr(flags & DRAW_CULLING)
                continue;
            else
                rasterize<true>(mem.framebuffer, triangle, prog, si);
        } else
            rasterize<false>(mem.framebuffer, triangle, prog, si);

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

// TODO: optimize
template<bool backface>
static inline void rasterize(
    Frame &frame,
    OutVertex *triangle,
    const Program &prog,
    const ShaderInterface &si
) {
    // viewport transform into a triangle
    float xm = frame.width / 2.f;
    float ym = frame.height / 2.f;

    Triangle t{
        glm::vec4{
            (triangle[0].gl_Position.x + 1) * xm,
            (triangle[0].gl_Position.y + 1) * ym,
            triangle [0].gl_Position.z,
            triangle [0].gl_Position.w,
        },
        glm::vec4{
            (triangle[1].gl_Position.x + 1) * xm,
            (triangle[1].gl_Position.y + 1) * ym,
            triangle [1].gl_Position.z,
            triangle [1].gl_Position.w,
        },
        glm::vec4{
            (triangle[2].gl_Position.x + 1) * xm,
            (triangle[2].gl_Position.y + 1) * ym,
            triangle [2].gl_Position.z,
            triangle [2].gl_Position.w,
        },
    };

    // prepare for rasterization
    t.get_area<backface>();
    bool failed;
    FragmentContext fc{ t, triangle, frame, prog, si, failed };

    if (failed)
        return;

    for (glm::uint y = fc.bl.y; y <= fc.tr.y; ++y) {
        fc.eval_at(fc.bl.x + .5f, y + .5f);
        for (glm::uint x = fc.bl.x; x <= fc.tr.x; ++x) {
            if (fc.should_draw<backface>()) {
                fc.draw();
            }

            fc.add_x();
        }
    }
}

static inline uint32_t to_rgba(const glm::vec4 color) {
    const uint8_t comp[] = {
        static_cast<uint8_t>(color.r * 255),
        static_cast<uint8_t>(color.g * 255),
        static_cast<uint8_t>(color.b * 255),
        static_cast<uint8_t>(color.a * 255),
    };

    return *reinterpret_cast<const uint32_t *>(comp);
}

static inline float triangle_area(glm::vec2 a, glm::vec2 b, glm::vec2 c) {
    return std::abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2;
}

/* (backface = clockwise, frontface = counterclockwise)
 * The triangle explained:
 *   A_
 *   |\
 * ab| \ca
 *   |  \
 *   V-->\
 *   B bc C
 *
 * Counterclockwise vectors:
 * ab = B - A
 * bc = C - B
 * ca = A - C
 *
 * The equation for line going through the sides of the triangles so that
 * when a point is evaluated on their position, it is positive when inside
 * the triangle (negative when the triangle is backface)
 *   AB: ab.x * (y - a.y) - ab.y * (x - a.x) = 0
 *   BC: bc.x * (y - b.y) - bc.y * (x - b.x) = 0
 *   CA: ca.x * (y - c.y) - bc.y * (x - c.x) = 0
 */

inline Triangle::Triangle(glm::vec4 a, glm::vec4 b, glm::vec4 c)
: a(a), b(b), c(c) {
    ab = glm::vec2{ b.x - a.x, b.y - a.y };
    bc = glm::vec2{ c.x - b.x, c.y - b.y };
    ca = glm::vec2{ a.x - c.x, a.y - c.y };
}

inline bool Triangle::is_backface() const {
    /* This is derived from the equation for BC evaluated at the point A.
     * The equation is negative when the triangle is backface.
     * The equation is 0 when the points are in line (don't form triangle)
     *   bc.x * (y - b.y) - bc.y * (x - b.x) <= 0        \ (x, y) = A
     *   bc.x * (a.y - b.y) - bc.y * (a.x - b.x) <= 0
     *   bc.x * (a.y - b.y) <= bc.y * (a.x - b.x)        \ a - b = -ab
     *   bc.x * (-ab.y) <= bx.y * (-ab.x)
     *   bx.x * ab.y >= bx.y * ab.x
     */
    return bc.x * ab.y >= bc.y * ab.x;
}

template<bool backface>
inline float Triangle::get_area() {
    /* This is derived from the fact that the area of parallelogram is:
     *     A_________D
     *     /        /
     *    /        /
     *   /________/
     *   B        C
     *
     * the absolute value of determinant:
     *   ||A.x - B.x  C.x - B.x||
     *   ||A.y - B.y  C.y - B.y||
     *
     * And the area of triangle ABC is just half of this area. The differences
     * are already precomputed:
     *   a.x - b.x = ab.x
     *   ...
     *
     * so:
     *   S = ||-ab.x  -bc.x||
     *       ||-ab.y  -bc.y||
     *   S = |(-ab.x) * (-bc.y) - (-ab.y) * (-bc.x)|
     *   S = |ab.x * bc.y - ab.y * bc.x|
     *
     * And when we know the orientation of the triangle, we cane eliminate the
     * absolute value with condition:
     *   S = ab.x * bc.y - ab.y * bc.x    when not backface
     *   S = ab.y * bc.x - ab.x * bc.y    when backface
     *
     * And the area needs to be divided by 2 to get the area of the triangle
     * and not the parallelogram.
     */
    if constexpr(backface)
        return area = (ab.y * bc.x - ab.x * bc.y) / 2;
    else
        return area = (ab.x * bc.y - ab.y * bc.x) / 2;
}

inline FragmentContext::FragmentContext(
    const Triangle &t,
    const OutVertex *vert,
    Frame &frame,
    const Program &prog,
    const ShaderInterface &si,
    bool &failed
) : t(t),
    prog(prog),
    vert(vert),
    color(reinterpret_cast<uint32_t *>(frame.color)),
    si(si),
    frame(frame)
{
    failed = false;

    // calculate bounding box
    glm::vec2 bl{ // bottom left
        std::max(0.f, std::min({t.a.x, t.b.x, t.c.x})),
        std::max(0.f, std::min({t.a.y, t.b.y, t.c.y})),
    };
    glm::vec2 tr{ // top right
        std::min<float>(frame.width - 1, std::max({t.a.x, t.b.x, t.c.x})),
        std::min<float>(frame.height - 1, std::max({t.a.y, t.b.y, t.c.y})),
    };

    if (bl.x >= tr.x || bl.y >= tr.y) {
        failed = true;
        return;
    }

    // make it integers
    this->bl = bl;
    this->tr = tr;

    // evaluate at initial (bottom left) position
    eval_at(this->bl.x + .5f, this->bl.y + .5f);
}

inline void FragmentContext::eval_at(const float x, const float y) {
    pt = glm::vec2{ x, y };

    // the subtraction is reused so that the compiler may optimize it
    abv = t.ab.x * (y - t.b.y) - t.ab.y * (x - t.b.x);
    bcv = t.bc.x * (y - t.b.y) - t.bc.y * (x - t.b.x);
    cav = t.ca.x * (y - t.c.y) - t.ca.y * (x - t.c.x);
}

/* these relations are derived from the side equations, example for side AB:
 *   f(x, y) = t.ab.x * (y - t.a.y) - t.ab.y * (x - t.a.x)
 *
 * Now change x by one:
 *   f(x + 1, y) = t.ab.x * (y - t.a.y) - t.ab.y * (x + 1 - t.a.x)
 *   f(x + 1, y) = t.ab.x * (y - t.a.y) - t.ab.y * (x - t.a.x) + (-t.ab.y)
 *   f(x + 1, y) = f(x, y) - t.ab.y
 *
 * The same can be done for f(x - 1, y), f(x, y + 1) and f(x, y - 1):
 *   f(x - 1, y) = f(x, y) + t.ab.y
 *   f(x, y + 1) = f(x, y) + t.ab.x
 *   f(x, y - 1) = f(x, y) - t.ab.x
 */

inline void FragmentContext::add_x() {
    // update the point
    pt.x += 1;

    // update the sides
    abv -= t.ab.y;
    bcv -= t.bc.y;
    cav -= t.ca.y;
}

inline void FragmentContext::sub_x() {
    // update the point
    pt.x -= 1;

    // update point evaluation
    abv += t.ab.y;
    bcv += t.bc.y;
    cav += t.ca.y;
}

inline void FragmentContext::add_y() {
    // update the point
    pt.y += 1;

    // update point evaluation
    abv += t.ab.x;
    bcv += t.bc.x;
    cav += t.ca.x;
}
inline void FragmentContext::sub_y() {
    // update the point
    pt.y -= 1;

    // update point evaluation
    abv -= t.ab.x;
    bcv -= t.bc.x;
    cav -= t.ca.x;
}

inline void FragmentContext::draw() {
    // calculate barycentric coordinates
    float m = 1 / t.area;
    glm::vec3 bc{
        triangle_area(t.b, t.c, pt) * m,
        triangle_area(t.a, t.c, pt) * m,
        0,
    };
    bc.z = 1 - bc.x - bc.y;

    InFragment in{
        .gl_FragCoord{
            pt.x,
            pt.y,
            t.a.z * bc.x + t.b.z * bc.y + t.c.z * bc.z,
            1.f,
        },
    };

    // get perspective adjusted coordinates
    float s = bc.x / t.a.w + bc.y / t.b.w + bc.z / t.c.w;
    glm::vec3 pbc{
        bc.x / (t.a.w * s),
        bc.y / (t.b.w * s),
        bc.z / (t.c.w * s),
    };

    // resolve the attributes
    for (size_t i = 0; i < maxAttributes; ++i) {
        // skip attribute
        AttributeType t = prog.vs2fs[i];
        if (t == AttributeType::EMPTY)
            continue;

        // copy integers from the first vertex
        int ti = static_cast<int>(t);
        if (ti > 8) {
            std::copy_n(
                reinterpret_cast<const uint32_t *>(vert->attributes + i),
                ti & 3,
                reinterpret_cast<int32_t *>(in.attributes + i)
            );
            continue;
        }

        // iterpolate template attributes
        for (int j = 0; j < ti; ++j) {
            in.attributes[i].v4[j] =
                vert[0].attributes[i].v4[j] * pbc.x +
                vert[1].attributes[i].v4[j] * pbc.y +
                vert[2].attributes[i].v4[j] * pbc.z;
        }
    }

    // call the shader
    OutFragment out;
    prog.fragmentShader(out, in, si);
    size_t p = (size_t)pt.y * frame.width + (size_t)pt.x;
    /*if (p >= frame.width * frame.height)
        0;*/
    color[(size_t)pt.y * frame.width + (size_t)pt.x] =
        to_rgba(out.gl_FragColor);
}

template<bool backface>
inline bool FragmentContext::should_draw() const {
    if constexpr(backface)
        return abv <= 0 && bcv <= 0 && cav <= 0;
    else
        return abv >= 0 && bcv >= 0 && cav >= 0;
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

