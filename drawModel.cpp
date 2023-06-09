/*!
 * @file
 * @brief This file contains functions for model rendering
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 *
 * Implemented by: Jakub Antonín Štigelr
 */
#include <student/drawModel.hpp>
#include <student/gpu.hpp>
#include <iostream>

///\endcond
void drawModel_vertexShader(
    OutVertex &outVertex,
    InVertex const &inVertex,
    ShaderInterface const &si
);

void drawModel_fragmentShader(
    OutFragment &outFragment,
    InFragment const &inFragment,
    ShaderInterface const &si
);

void prepare_nodes(
    GPUMemory &mem,
    CommandBuffer &cb,
    Model const &model
);

static inline glm::vec4 read_texture_inline(
    Texture const &texture,
    glm::vec2 uv
);

/**
 * @brief This function prepares model into memory and creates command buffer
 *
 * @param mem gpu memory
 * @param commandBuffer command buffer
 * @param model model structure
 */
//! [drawModel]
void prepareModel(
    GPUMemory &mem,
    CommandBuffer &commandBuffer,
    Model const &model
) {
    // set the memory
    std::copy(model.buffers.begin(), model.buffers.end(), mem.buffers);
    mem.programs->fragmentShader = drawModel_fragmentShader;
    mem.programs->vertexShader = drawModel_vertexShader;
    std::copy(model.textures.begin(), model.textures.end(), mem.textures);
    mem.programs[0].vs2fs[0] = AttributeType::VEC3;
    mem.programs[0].vs2fs[1] = AttributeType::VEC3;
    mem.programs[0].vs2fs[2] = AttributeType::VEC2;
    mem.programs[0].vs2fs[3] = AttributeType::UINT;

    pushClearCommand(commandBuffer, glm::vec4{ .1, .15, .1, 1 });

    prepare_nodes(mem, commandBuffer, model);
}
//! [drawModel]

/**
 * @brief This function represents vertex shader of texture rendering method.
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param si shader interface
 */
//! [drawModel_vs]
void drawModel_vertexShader(
    OutVertex &outVertex,
    InVertex const &inVertex,
    ShaderInterface const &si
) {
    // extract attributes
    const glm::vec3 &pos = inVertex.attributes->v3;
    const glm::vec3 &norm = inVertex.attributes[1].v3;
    const glm::vec2 &tex = inVertex.attributes[2].v2;

    // extract uniforms
    const glm::mat4 &view = si.uniforms->m4;
    const glm::mat4 &model = si.uniforms[5 * inVertex.gl_DrawID + 10].m4;
    const glm::mat4 &itmod = si.uniforms[5 * inVertex.gl_DrawID + 11].m4;

    // set position
    auto mpos = model * glm::vec4(pos, 1.f);
    outVertex.gl_Position = view * mpos;

    // set attributes
    outVertex.attributes->v3 = mpos;
    outVertex.attributes[1].v3 = itmod * glm::vec4(norm, 0.f);
    outVertex.attributes[2].v2 = tex;
    outVertex.attributes[3].u1 = inVertex.gl_DrawID;
}
//! [drawModel_vs]

/**
 * @brief This functionrepresents fragment shader of texture rendering method.
 *
 * @param outFragment output fragment
 * @param inFragment input fragment
 * @param si shader interface
 */
//! [drawModel_fs]
void drawModel_fragmentShader(
    OutFragment &outFragment,
    InFragment const &inFragment,
    ShaderInterface const &si
) {
    // extract the attributes
    const glm::vec3 &pos = inFragment.attributes->v3;
    glm::vec3 norm = inFragment.attributes[1].v3;
    const glm::vec2 &tex = inFragment.attributes[2].v2;
    uint32_t draw_id = inFragment.attributes[3].u1;

    // extract uniforms
    const glm::vec3 &lpos = si.uniforms[1].v3;
    const glm::vec3 &cpos = si.uniforms[2].v3;
    const glm::vec4 &dcol = si.uniforms[5 * draw_id + 12].v4;
    int32_t tex_id = si.uniforms[5 * draw_id + 13].i1;
    bool ds = si.uniforms[5 * draw_id + 14].v1 != 0;

    // normalize the mormal
    auto nor = glm::normalize(norm);

    auto col = tex_id >= 0
        ? read_texture_inline(si.textures[tex_id], tex)
        : dcol;

    if (ds && glm::dot(nor, pos - cpos) > 0)
        nor = -nor;

    glm::vec3 col3 = col;

    outFragment.gl_FragColor = glm::vec4((glm::clamp(
        glm::dot(glm::normalize(lpos - pos), nor),
        0.f,
        1.f
    ) + .2f) * col3, col.a);
}
//! [drawModel_fs]
void prepare_nodes(
    GPUMemory &mem,
    CommandBuffer &cb,
    Model const &model
) {
    // use stacks on heap instead of recursion

    // copy the nodes to prevent modifying the original vector
    std::vector<Node> nodes{ model.roots.rbegin(), model.roots.rend() };
    // contains matrixes
    std::vector<glm::mat4> mats = { glm::mat4{ 1 } };
    // contains pop indexes for matrixes
    std::vector<size_t> pids = { 0 };

    size_t id = SIZE_MAX;
    while (nodes.size()) {
        while (pids.back() >= nodes.size()) {
            pids.pop_back();
            mats.pop_back();
        }

        const Node node = std::move(nodes.back());
        nodes.pop_back();

        // minimize the stacks
        if (nodes.size() == pids.back()) {
            mats.back() *= node.modelMatrix;
        } else {
            mats.push_back(mats.back() * node.modelMatrix);
            pids.push_back(nodes.size());
        }

        if (node.mesh >= 0) {
            ++id;
            const Mesh &mesh = model.meshes[node.mesh];
            static size_t cnt = 0;

            mem.uniforms[5 * id + 10].m4 = mats.back();
            mem.uniforms[5 * id + 11].m4 =
                glm::transpose(glm::inverse(mats.back()));
            mem.uniforms[5 * id + 12].v4 = mesh.diffuseColor;
            mem.uniforms[5 * id + 13].i1 = mesh.diffuseTexture;
            mem.uniforms[5 * id + 14].v1 = mesh.doubleSided;

            pushDrawCommand(
                cb,
                mesh.nofIndices,
                0,
                VertexArray{
                    .vertexAttrib {
                        mesh.position,
                        mesh.normal,
                        mesh.texCoord,
                    },
                    .indexBufferID = mesh.indexBufferID,
                    .indexOffset = mesh.indexOffset,
                    .indexType = mesh.indexType,
                },
                !mesh.doubleSided
            );
        }

        nodes.insert(
            nodes.end(),
            node.children.rbegin(),
            node.children.rend()
        );
    }
}

static inline glm::vec4 read_texture_inline(
    Texture const &texture,
    glm::vec2 uv
) {
    // the inline version is much faster

    if (!texture.data)
        return glm::vec4(0.f);

    size_t x = (uv.x - std::floor(uv.x)) * (texture.width - 1) + .5f;
    size_t y = (uv.y - std::floor(uv.y)) * (texture.height - 1) + .5f;
    glm::vec4 color{ 0.f, 0.f, 0.f, 1.f };

    const uint8_t *tex =
        texture.data + (y * texture.width + x) * texture.channels;

    switch (texture.channels) {
    case 4:
        color.a = tex[3] / 255.f;
    case 3:
        color.b = tex[2] / 255.f;
    case 2:
        color.g = tex[1] / 255.f;
    case 1:
        color.r = *tex / 255.f;
    default:
        break;
    }

    return color;
}
