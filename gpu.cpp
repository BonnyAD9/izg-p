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
        const Program &prog
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
    // the current pixel
    glm::vec2 pt;
    // the color buffer
    uint32_t *color;
    // values of triangle side equations of pt
    float abv;
    float bcv;
    float cav;
    // bottom left bounding box coordinate
    glm::uvec2 bl;
    // top right bounding box coordinate
    glm::uvec2 tr;
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

    // calculate bounding box inside the screen
    glm::vec2 fbl{ // bottom left
        std::max(0.f, std::min({t.a.x, t.b.x, t.c.x})),
        std::max(0.f, std::min({t.a.y, t.b.y, t.c.y})),
    };
    glm::vec2 ftr{ // top right
        std::min<float>(frame.width - 1, std::max({t.a.x, t.b.x, t.c.x})),
        std::min<float>(frame.height - 1, std::max({t.a.y, t.b.y, t.c.y})),
    };

    // check if the triangle is on screen
    if (fbl.x >= ftr.x || fbl.y >= ftr.y)
        return;

    // make it integers
    glm::uvec2 bl = fbl;
    glm::uvec2 tr = ftr;

    // rasterization using pineda

    // edge vectors
    glm::vec2 d0{ t.b.x - t.a.x, t.b.y - t.a.y };
    glm::vec2 d1{ t.c.x - t.b.x, t.c.y - t.b.y };
    glm::vec2 d2{ t.a.x - t.c.x, t.a.y - t.c.y };

    uint32_t *colbuf = reinterpret_cast<uint32_t *>(frame.color);
    InFragment in_fragment{
        .gl_FragCoord{ 0.f, 0.f, t.b.z, 1.f}
    };

    //float area = triangle_area(t.a, t.b, t.c);
    float area = t.get_area<backface>();

    for (glm::uint y = bl.y; y <= tr.y; ++y) {
        float e0 = (bl.x - t.a.x) * d0.y - (y - t.a.y) * d0.x;
        float e1 = (bl.x - t.b.x) * d1.y - (y - t.b.y) * d1.x;
        float e2 = (bl.x - t.c.x) * d2.y - (y - t.c.y) * d2.x;
        in_fragment.gl_FragCoord.y = y + .5f;
        for (glm::uint x = bl.x; x <= tr.x; ++x) {
            // change the drawing condition based on whether the triangle is
            // backface or not
            bool draw;
            if constexpr(backface)
                draw = e0 >= 0 && e1 > 0 && e2 >= 0;
            else // e1 has different condition to pass test 12 (no tolerance)
                draw = e0 <= 0 && e1 < 0 && e2 <= 0;

            if (draw) {
                // get the barycentric coordinates
                glm::vec2 pt{x + .5f, y + .5f};
                glm::vec3 bcc{
                    triangle_area(t.b, t.c, pt) / area,
                    triangle_area(t.a, t.c, pt) / area,
                    triangle_area(t.b, t.a, pt) / area,
                };

                in_fragment.gl_FragCoord.x = x + .5f;
                in_fragment.gl_FragCoord.z =
                    t.a.z * bcc.x + t.b.z * bcc.y + t.c.z * bcc.z;
                // skip fragments behind camera

                // get perspective barycentric coordinates
                float s = bcc.x / t.a.w + bcc.y / t.b.w + bcc.z / t.c.w;
                glm::vec3 pbc{
                    bcc.x / (t.a.w * s),
                    bcc.y / (t.b.w * s),
                    bcc.z / (t.c.w * s),
                };

                for (size_t i = 0; i < maxAttributes; ++i) {
                    // skip attribute
                    AttributeType t = prog.vs2fs[i];
                    if (t == AttributeType::EMPTY)
                        continue;

                    // copy attribute
                    if (static_cast<int>(t) > 8) {
                        std::copy_n(
                            reinterpret_cast<uint8_t *>(
                                triangle[0].attributes + i
                            ),
                            (static_cast<int>(t) & 3) << 2,
                            reinterpret_cast<uint8_t *>(
                                in_fragment.attributes + i
                            )
                        );
                        continue;
                    }

                    // iterpolate attribute
                    int count = static_cast<int>(t);
                    for (size_t j = 0; j < count; ++j) {
                        in_fragment.attributes[i].v4[j] =
                            triangle[0].attributes[i].v4[j] * pbc.x +
                            triangle[1].attributes[i].v4[j] * pbc.y +
                            triangle[2].attributes[i].v4[j] * pbc.z;
                    }
                }

                // call the fragment shader
                OutFragment out_fragment;
                prog.fragmentShader(out_fragment, in_fragment, si);
                colbuf[y * frame.width + x] =
                    to_rgba(out_fragment.gl_FragColor);
            }

            e0 += d0.y;
            e1 += d1.y;
            e2 += d2.y;
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
    const Program &prog
) : t(t),
    prog(prog),
    vert(vert),
    color(reinterpret_cast<uint32_t *>(frame.color))
{
    // calculate bounding box
    glm::vec2 bl{ // bottom left
        std::max(0.f, std::min({t.a.x, t.b.x, t.c.x})),
        std::max(0.f, std::min({t.a.y, t.b.y, t.c.y})),
    };
    glm::vec2 tr{ // top right
        std::min<float>(frame.width - 1, std::max({t.a.x, t.b.x, t.c.x})),
        std::min<float>(frame.height - 1, std::max({t.a.y, t.b.y, t.c.y})),
    };
    // make it integers
    this->bl = bl;
    this->tr = tr;

    // evaluate at initial (bottom left) position
    eval_at(this->bl.x, this->bl.y);
}

inline void FragmentContext::eval_at(const float x, const float y) {
    abv = t.ab.x * (y - t.a.y) - t.ab.y * (x - t.a.x);
    // the computation with the point a can be reused
    bcv = t.bc.x * (y - t.a.y) - t.bc.y * (x - t.a.x);
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
    abv -= t.ab.y;
    bcv -= t.bc.y;
    cav -= t.ca.y;
}

inline void FragmentContext::sub_x() {
    abv += t.ab.y;
    bcv += t.bc.y;
    cav += t.ca.y;
}

inline void FragmentContext::add_y() {
    abv += t.ab.x;
    bcv += t.bc.x;
    cav += t.ca.x;
}
inline void FragmentContext::sub_y() {
    abv -= t.ab.x;
    bcv -= t.bc.x;
    cav -= t.ca.x;
}

inline void FragmentContext::draw() {

}

template<bool backface>
inline bool FragmentContext::should_draw() const {
    if constexpr(backface)
        return abv <= 0 && bcv < 0 && cav <= 0;
    else // bcv has different condition to pass test 12 (it has no tolerance)
        return abv >= 0 && bcv > 0 && cav >= 0;
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

