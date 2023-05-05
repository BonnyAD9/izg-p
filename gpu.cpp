/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 *
 * Implemented by: Jakub Antonín Štigler (xstigl00)
 */

#include <student/gpu.hpp>
#include <algorithm>
#include <iostream>

// used to extract attributes for vertex shader for faster iteration
struct VExtAttrib {
    inline VExtAttrib(const VertexAttrib *attributes, const Buffer *buffers);
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

// used to extract attributes for fragment shaders
// simillar to VExtAttrib
struct FExtAttrib {
    inline FExtAttrib(const AttributeType *types);
    inline void set_attrib(Attribute *out_attribs, glm::vec3 pbc) const;
    inline void set_arrs(const OutVertex *vout);

    // copied attributes
    size_t ind[maxAttributes];
    size_t siz[maxAttributes];
    const uint32_t *arr[maxAttributes];
    size_t cnt;

    // interpolated attributes
    size_t iind[maxAttributes];
    size_t isiz[maxAttributes];
    const float *iarr[3][maxAttributes];
    size_t icnt;
};

// contains the points of the triangle and its precalculated edges
struct Triangle {
    inline Triangle(glm::vec4 a, glm::vec4 b, glm::vec4 c);
    // does viewport transform
    inline void to_viewport(size_t width, size_t height);
    // the area and side vectors are precomputed optionally
    inline float get_area();

    // points of the triangle
    glm::vec4 a;
    glm::vec4 b;
    glm::vec4 c;

    // vectors of the triangle sides (in xy 2D)
    // used in many formulas -> they are precomputed here
    // ab = vector from a to b
    // ...
    glm::vec2 ab;
    glm::vec2 bc;
    glm::vec2 ca;

    // area of the triangle * 2 (may be negative)
    float area;
};

// used for rasterizing triangles
struct Rasterizer {
    inline Rasterizer(
        const Triangle &t,
        const OutVertex *vert,
        Frame &frame,
        const Program &prog,
        const ShaderInterface &si,
        FExtAttrib &fat,
        bool &failed
    );
    // evaluates the equations at the given point
    inline void eval_at(const float x, const float y);
    // changes the evaluated point by 1 in the x axis
    inline void add_x();
    // changes the evaluated point by -1 in the x axis
    inline void sub_x();
    // changes the evaluated point by 1 in the y axis
    inline void add_y();
    // changes the evaluated point by -1 in the y axis
    inline void sub_y();
    // calls fragment shader and draws the current point
    inline void draw();
    // returns true if the triangle should be drawn, the template parameter
    // is used for optimization
    inline bool should_draw() const;
    inline void save_pos();
    inline void load_pos();
    // functions that draw lines from the current position
    // they return true if they stop on pixel (not bounding box)
    inline bool draw_right();
    inline bool draw_left();
    inline bool skip_right();
    inline bool skip_left();
    // move up with both px and pt, return true if this is outside of bounding
    // box, if this returns false, px and pt are different
    inline bool move_up();

    // tirangle to draw
    const Triangle &t;
    // the results of vertex shader
    const OutVertex *vert;
    // the shader program
    const Program &prog;
    // the constants for shader
    const ShaderInterface &si;
    // the extracted attributes
    const FExtAttrib &fat;
    // the frame buffer
    Frame &frame;
    // the current pixel (float coordinates)
    glm::vec2 pt;
    // the current pixel (int coordinates)
    // it is not synchronized with the pt, but should be when calling the draw
    // function
    glm::ivec2 px;
    // the color buffer
    uint32_t *color;
    // values of triangle side equations of pt
    // thans to the transformed side vectors of the triangle t
    // these are also equal to the barycentric coordinates at poin pt
    float abv;
    float bcv;
    float cav;

    // save and restore variables
    glm::vec2 spt;
    glm::ivec2 spx;
    float sabv;
    float sbcv;
    float scav;

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

// craetes the Rasterizer and rasterizes the given triangle
static inline void rasterize(
    Frame &frame,
    Triangle t,
    const OutVertex *vert,
    const Program &prog,
    const ShaderInterface &si,
    FExtAttrib &fat
);

static inline uint32_t to_rgba(glm::vec4 color);

static inline glm::vec4 from_rgba(const uint32_t color);

static inline void clip_near_and_rasterize(
    Frame &frame,
    Triangle t,
    const OutVertex *vert,
    const Program &prog,
    const ShaderInterface &si,
    FExtAttrib &fat
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
    // here are determined things like whether the culling is enabled
    // so that they can be used as a compile time if expression later in the
    // code
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
    VExtAttrib vat{ cmd.vao.vertexAttrib, mem.buffers };
    FExtAttrib fat{ prog.vs2fs };

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
            vat.set_attrib(in_vertex.gl_VertexID, in_vertex.attributes);
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

        // skip backface triangles if culling is enabled
        if constexpr(flags & DRAW_CULLING) {
            // area is negative when backface
            if (t.get_area() <= 0)
                continue;
        }

        rasterize(mem.framebuffer, t, triangle, prog, si, fat);
    }
}

inline VExtAttrib::VExtAttrib(
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

inline void VExtAttrib::set_attrib(size_t index, Attribute *out_attribs) const {
    for (size_t j = 0; j < cnt; ++j) {
        std::copy_n(
            arr[j] + (index * str[j]),
            siz[j],
            reinterpret_cast<char *>(out_attribs + ind[j])
        );
    }
}

// TODO: optimize (
//   make the code readable
// )
static inline void rasterize(
    Frame &frame,
    Triangle t,
    const OutVertex *vert,
    const Program &prog,
    const ShaderInterface &si,
    FExtAttrib &fat
) {
    t.to_viewport(frame.width, frame.height);

    // prepare for rasterization
    bool failed;
    Rasterizer fc{ t, vert, frame, prog, si, fat, failed };

    if (failed)
        return;

    // the following code is quite complicated, but the idea is simple
    // when you rasterize the triangle you don't have to check every pixel
    // when you already find one pixel with the triangle, you can jsut check
    // the neighbouring pixels.
    // this algorithm searches for the triangle from the bottom left and
    // once it finds the triangle, it searches from left to right while
    // staying inside the triangle:
    // _________
    // \<<<<<<<|
    //  \<>>>>>|
    //   \<<<<<|
    //    \<>>>|
    //     \<<<|
    //      \<>|
    //       \<|
    // >>>>>>>\|
    //
    // in reality the algorithm is much more complicated because of all the
    // edge cases (such as triangles that skip some lines):
    //     *
    //     **
    //      *
    //
    //       *
    //
    //        *

    // comment legend:
    // ?   unknown position
    // > < empty position traversed in direction
    // *   filled position
    // +   current filled position
    // /   current empty position
    // .   current unknown position
#define fc_move_up() if (!fc.move_up()) return;
    // this first part is for finding the triangle
    do {
        do {
            // ????????????
            if (fc.should_draw() || fc.skip_right()) {
                // >>>>*???????
                fc.draw();
                fc.draw_right();
                // >>>>+++*----

                fc_move_up();
                // ???????.????
                // >>>>****----

                if (fc.should_draw()) {
                    fc.draw();
                    fc.save_pos();
                    fc.draw_right();
                    fc.load_pos();
                    fc.draw_left();
                    // --+*******--
                    // >>>>****----
                    break;
                }
                // ???????/????
                // >>>>****----

                fc.save_pos();
                if (fc.skip_left()) {
                    fc.draw();
                    fc.draw_left();
                    // --+***<<----
                    // >>>>****----
                    break;
                }
                fc.load_pos();
                // -------/????
                // >>>>****----

                if (fc.skip_right()) {
                    fc.draw();
                    fc.save_pos();
                    fc.draw_right();
                    fc.load_pos();
                    // ------->>+*-
                    // >>>>****----
                    break;
                }
                // ------->>>>/
                // >>>>****----

                fc_move_up();
                // ???????????.
                // ------->>>>/
                // >>>>****----

                if (fc.should_draw() || fc.skip_left()) {
                    fc.draw();
                    fc.draw_left();
                    // ----+***<<<<
                    // ------->>>>/
                    // >>>>****----
                    break;
                }

                // /<<<<<<<<<<<
                // ------->>>>/
                // >>>>****----
                continue;
            }
            // >>>>>>>>>>>/

            fc_move_up();
            // ???????????.
            // >>>>>>>>>>>>

            if (fc.should_draw() || fc.skip_left()) {
                fc.draw();
                fc.draw_left();
                // ----+***<<<<
                // >>>>>>>>>>>>
                break;
            }
            // /<<<<<<<<<<<
            // >>>>>>>>>>>>
        } while (fc.move_up());
        // ----+***----

        while (fc.move_up()) {
            // ????.???????
            // ----****----

            if (fc.should_draw()) {
                fc.draw();
                fc.save_pos();
                fc.draw_left();
                fc.load_pos();
                fc.draw_right();
                // --******+???
                // ----****----
                continue;
            } else {
                // ????/???????
                // ----****----
                fc.save_pos();
                if (fc.skip_left()) {
                    fc.draw();
                    fc.draw_left();
                    // -+*<<-------
                    // ----****----
                    continue;
                }
                fc.load_pos();
                // ----/???????
                // ----****----

                if (!fc.skip_right()) {
                    fc_move_up();
                    // ???????????.
                    // ---->>>>>>>>
                    // ----****----

                    if (fc.should_draw() || fc.skip_left()) {
                        fc.draw();
                        fc.draw_left();
                        // ----+***<<<<
                        // ---->>>>>>>>
                        // ----****----
                        continue;
                    }

                    // /<<<<<<<<<<<
                    // ---->>>>>>>>
                    // ----****----
                    break;
                }
                // ---->>>+????
                // ----****----

                fc.draw();
                fc.draw_right();
                // ---->>>**+--
                // ----****----
            }
            // ----***+----

            fc_move_up();
            // ???????.????
            // ----****----

            if (fc.should_draw()) {
                fc.draw();
                fc.save_pos();
                fc.draw_right();
                fc.load_pos();
                fc.draw_left();
                // --+*******--
                // ----****----
                continue;
            }
            // ???????/????
            // ----****----

            fc.save_pos();
            if (fc.skip_right()) {
                fc.draw();
                fc.save_pos();
                fc.draw_right();
                fc.load_pos();
                // ------->>+*-
                // ----****----
                continue;
            }
            fc.load_pos();
            // ???????/----
            // ----****----

            if (!fc.skip_left()) {
                // /-----------
                // ----****----
                break;
            }
            // ?????+<<----
            // ----****----

            fc.draw();
            fc.draw_left();
            // ??+***<<----
            // ----****----
        }
    } while (fc.move_up());
#undef fc_move_up
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

static inline glm::vec4 from_rgba(const uint32_t color) {
    const uint8_t *comp = reinterpret_cast<const uint8_t *>(&color);
    return glm::vec4{
        comp[0] / 255.f,
        comp[1] / 255.f,
        comp[2] / 255.f,
        comp[3] / 255.f,
    };
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
 *
 * The magic behind the tirangle is that the side vectors are divided
 * by the area. This simplifies almost every computation with the triangle.
 * Namely the barycentric coordinates and it also removes the difference
 * between backside and front side triangles so there doesn't need to be
 * any special cases for the two types of triangles.
 */

inline Triangle::Triangle(glm::vec4 a, glm::vec4 b, glm::vec4 c)
: a(a), b(b), c(c) {
    ab = glm::vec2{ b.x - a.x, b.y - a.y };
    bc = glm::vec2{ c.x - b.x, c.y - b.y };
    ca = glm::vec2{ a.x - c.x, a.y - c.y };

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
     * and not the parallelogram. This is however not done becouse in other
     * calculations the 2 calncels out.
     *
     * The cool part is that if we don't calculate the absolute value, the sign
     * tells us whether the triangle is backface or not
     */
    area = ab.x * bc.y - ab.y * bc.x;
}

inline void Triangle::to_viewport(size_t width, size_t height) {
    float xm = width / 2.f;
    float ym = height / 2.f;

    // transform x and y
    a.x = (a.x + 1) * xm;
    a.y = (a.y + 1) * ym;
    b.x = (b.x + 1) * xm;
    b.y = (b.y + 1) * ym;
    c.x = (c.x + 1) * xm;
    c.y = (c.y + 1) * ym;

    // update the sides
    float am = 1 / get_area();

    // transform the sides so that when calculating the value of the triangle
    // side equations, it is equal to the barycentric coordinates
    ab *= am;
    bc *= am;
    ca *= am;

    // transform division to multiplication
    a.w = 1 / a.w;
    b.w = 1 / b.w;
    c.w = 1 / c.w;
}

inline float Triangle::get_area() {
    ab.x = b.x - a.x;
    ab.y = b.y - a.y;
    bc.x = c.x - b.x;
    bc.y = c.y - b.y;
    ca.x = a.x - c.x;
    ca.y = a.y - c.y;

    // recalculate the area
    return area = ab.x * bc.y - ab.y * bc.x;
}

inline Rasterizer::Rasterizer(
    const Triangle &t,
    const OutVertex *vert,
    Frame &frame,
    const Program &prog,
    const ShaderInterface &si,
    FExtAttrib &fat,
    bool &failed
) : t(t),
    prog(prog),
    vert(vert),
    color(reinterpret_cast<uint32_t *>(frame.color)),
    si(si),
    frame(frame),
    fat(fat)
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
    //this->tr.x = std::ceil(tr.x);
    //this->tr.y = std::ceil(tr.y);
    px = bl;

    fat.set_arrs(vert);

    // evaluate at initial (bottom left) position
    eval_at(this->bl.x + .5f, this->bl.y + .5f);
}

inline void Rasterizer::eval_at(const float x, const float y) {
    pt = glm::vec2{ x, y };

    // the subtraction is reused so that the compiler may optimize it
    abv = t.ab.x * (y - t.b.y) - t.ab.y * (x - t.b.x);
    bcv = t.bc.x * (y - t.b.y) - t.bc.y * (x - t.b.x);
    cav = 1 - abv - bcv;
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

inline void Rasterizer::add_x() {
    // update the point
    pt.x += 1;

    // update the sides
    abv -= t.ab.y;
    bcv -= t.bc.y;
    cav -= t.ca.y;
}

inline void Rasterizer::sub_x() {
    // update the point
    pt.x -= 1;

    // update point evaluation
    abv += t.ab.y;
    bcv += t.bc.y;
    cav += t.ca.y;
}

inline void Rasterizer::add_y() {
    // update the point
    pt.y += 1;

    // update point evaluation
    abv += t.ab.x;
    bcv += t.bc.x;
    cav += t.ca.x;
}
inline void Rasterizer::sub_y() {
    // update the point
    pt.y -= 1;

    // update point evaluation
    abv -= t.ab.x;
    bcv -= t.bc.x;
    cav -= t.ca.x;
}

inline void Rasterizer::draw() {
    InFragment in{
        .gl_FragCoord{
            pt.x,
            pt.y,
            t.a.z * bcv + t.b.z * cav + t.c.z * abv,
            1.f,
        },
    };

    // get perspective adjusted coordinates
    float s = 1 / (bcv * t.a.w + cav * t.b.w + abv * t.c.w);
    glm::vec3 pbc{ bcv * t.a.w * s, cav * t.b.w * s, abv * t.c.w * s };

    fat.set_attrib(in.attributes, pbc);

    // call the shader
    OutFragment out;
    prog.fragmentShader(out, in, si);
    size_t p = px.y * frame.width + px.x;

    if (frame.depth[p] > in.gl_FragCoord.z) {
        if (out.gl_FragColor.a > .5f)
            frame.depth[p] = in.gl_FragCoord.z;
        color[p] = to_rgba(from_rgba(color[p]) * (1 - out.gl_FragColor.a)
            + out.gl_FragColor * out.gl_FragColor.a);
    }
}

inline bool Rasterizer::should_draw() const {
    return abv >= 0 && bcv >= 0 && cav >= 0;
}

inline void Rasterizer::save_pos() {
    spt = pt;
    spx = px;
    sabv = abv;
    sbcv = bcv;
    scav = cav;
}

inline void Rasterizer::load_pos() {
    pt = spt;
    px = spx;
    abv = sabv;
    bcv = sbcv;
    cav = scav;
}

// these funcions draw lines and ensure that both px and pt stay inside the
// bounding box of the triangle. They don't do anything with the current pixel
inline bool Rasterizer::draw_right() {
    for (++px.x; px.x <= tr.x; ++px.x) {
        add_x();
        if (!should_draw())
            return true;
        draw();
    }
    --px.x;
    return false;
}

inline bool Rasterizer::draw_left() {
    for (--px.x; px.x >= bl.x; --px.x) {
        sub_x();
        if (!should_draw())
            return true;
        draw();
    }
    ++px.x;
    return false;
}

inline bool Rasterizer::skip_right() {
    for (++px.x; px.x <= tr.x; ++px.x) {
        add_x();
        if (should_draw())
            return true;
    }
    --px.x;
    return false;
}


inline bool Rasterizer::skip_left() {
    for (--px.x; px.x >= bl.x; --px.x) {
        sub_x();
        if (should_draw())
            return true;
    }
    ++px.x;
    return false;
}

inline bool Rasterizer::move_up() {
    if (++px.y > tr.y)
        return false;
    add_y();
    return true;
}

inline FExtAttrib::FExtAttrib(const AttributeType *types)
    : cnt(0), icnt(0)
{
    for (size_t i = 0; i < maxAttributes; ++i) {
        if (types[i] == AttributeType::EMPTY)
            continue;

        int t = static_cast<int>(types[i]);

        // regular
        if (t > 8) {
            ind[cnt] = i;
            siz[cnt] = t & 3;
            ++cnt;
            continue;
        }

        // interpolated
        iind[icnt] = i;
        isiz[icnt] = t;
        ++icnt;
    }
}

inline void FExtAttrib::set_arrs(const OutVertex *vout) {
    // non interpolated
    for (size_t i = 0; i < cnt; ++i)
        arr[i] = reinterpret_cast<const uint32_t *>(vout->attributes + ind[i]);

    // interpolated
    for (size_t i = 0; i < icnt; ++i) {
        iarr[0][i] =
            reinterpret_cast<const float *>(vout[0].attributes + iind[i]);
        iarr[1][i] =
            reinterpret_cast<const float *>(vout[1].attributes + iind[i]);
        iarr[2][i] =
            reinterpret_cast<const float *>(vout[2].attributes + iind[i]);
    }
}

inline void FExtAttrib::set_attrib(
    Attribute *out_attribs,
    glm::vec3 pbc
) const {
    // non interpolated
    for (size_t i = 0; i < cnt; ++i) {
        std::copy_n(
            arr[i],
            siz[i],
            reinterpret_cast<uint32_t *>(out_attribs + ind[i])
        );
    }

    // interpolated
    for (size_t i = 0; i < icnt; ++i) {
        for (size_t j = 0; j < isiz[i]; ++j) {
            out_attribs[iind[i]].v4[j] =
                iarr[0][i][j] * pbc.x +
                iarr[1][i][j] * pbc.y +
                iarr[2][i][j] * pbc.z;
        }
    }
}

static inline void clip_near_and_rasterize(
    Frame &frame,
    Triangle t,
    const OutVertex *vert,
    const Program &prog,
    const ShaderInterface &si,
    FExtAttrib &fat
) {
    glm::vec4 pts[] = { t.a, t.b, t.c };
    size_t pc = 0;
    glm::vec4 clp[3];
    size_t cc = 0;

    // get clipped and unclipped triangles
    for (size_t i = 0; i < 3; ++i) {
        if (   -pts[i].w <= pts[i].x <= pts[i].w
            && -pts[i].w <= pts[i].y <= pts[i].w
            && -pts[i].w <= pts[i].z <= pts[i].w
        ) {
            pts[pc++] = pts[i];
            continue;
        }

        clp[cc++] = pts[i];
    }

    switch (cc) {
    case 0: // triangle is not clipped
        rasterize(frame, t, vert, prog, si, fat);
        return;
    case 1: { // one point is outside
        // TODO
    }
        return;
    case 2: { // two points are outside
        // TODO
    }
        return;
    case 3: // triangle is behind the camera
        return;
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

