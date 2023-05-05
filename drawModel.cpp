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
    (void)mem;
    (void)commandBuffer;
    (void)model;
    /// \todo Tato funkce připraví command buffer pro model a nastaví správně pamět grafické karty.<br>
    /// Vaším úkolem je správně projít model a vložit vykreslovací příkazy do commandBufferu.
    /// Zároveň musíte vložit do paměti textury, buffery a uniformní proměnné, které buffer command buffer využívat.
    /// Bližší informace jsou uvedeny na hlavní stránce dokumentace a v testech.

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
    (void)outVertex;
    (void)inVertex;
    (void)si;
    /// \todo Tato funkce reprezentujte vertex shader.<br>
    /// Vaším úkolem je správně trasnformovat vrcholy modelu.
    /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.
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
    (void)outFragment;
    (void)inFragment;
    (void)si;
    /// \todo Tato funkce reprezentujte fragment shader.<br>
    /// Vaším úkolem je správně obarvit fragmenty a osvětlit je pomocí lambertova osvětlovacího modelu.
    /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.
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

        ++id;
        const Node node = std::move(nodes.back());
        nodes.pop_back();

        mats.push_back(mats.back() * node.modelMatrix);
        pids.push_back(nodes.size());

        if (node.mesh >= 0) {
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

        nodes.insert(nodes.end(), node.children.rbegin(), node.children.rend());
    }
}

