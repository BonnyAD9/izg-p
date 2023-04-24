/*!
 * @file
 * @brief This file contains example structure of drawTriangle function
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include <student/fwd.hpp>

///\cond HIDDEN_SYMBOLS

//![gpu_execute_commands]

void clear(GPUMemory&mem,ClearCommand cmd){
  if(cmd.clearColor){
    float red = cmd.color.r;
    mem.framebuffer.color[0] = (uint8_t)red*255.f;
    //...
  }
}

void gpu_execute(GPUMemory&mem,CommandBuffer const&cb){
  for(uint32_t i=0;i<cb.nofCommands;++i){
    CommandType type = cb.commands[i].type;
    CommandData data = cb.commands[i].data;
    if(type == CommandType::CLEAR){clear(mem,data.clearCommand);
  }
}
//![gpu_execute_commands]

//![gpu_execute_before_vs]
void runVertexAssembly(){
  computeVertexID();
  readAttributes();
}

void draw(GPUMemory&mem,DrawCommand cmd){
  
  Program prg = mem.programs[cmd.programID];

  for(every vertex v < cmd.nofVertices){
    InVertex inVertex;
    OutVertex outVertex;
    runVertexAssembly(inVertex,cmd.vao,v);
    ShaderInterface si;
    prg.vertexShader(outVertex,inVertex,si);
  }
}

void gpu_execute(GPUMemory&mem,CommandBuffer const&cb){
  for(... commands ...){
    if (commandType == DRAW )
      draw(mem, drawCommand);
  }
}

//![gpu_execute_before_vs]



//![gpu_execute_after_vs]
void runVertexAssembly(){
  computeVertexID();
  readAttributes();
}

void runPrimitiveAssembly(primitive,VertexArray vao,t,Program prg){
  for(every vertex v in triangle){
    InVertex inVertex;
    runVertexAssembly(inVertex,vao,t+v);
    prg.vertexShader(primitive.vertex,inVertex,prg.uniforms);
  }
}

void rasterizeTriangle(frame,primitive,prg){
  for(pixels in frame){
    if(pixels in primitive){
      InFragment inFragment;
      createFragment(inFragment,primitive,barycentrics,pixelCoord,prg);
      OutFragment outFragment;
      prg.fragmentShader(outFragment,inFragment,uniforms);
    }
  }
}

void draw(GPUMemory&mem,DrawCommand const&cmd){
  for(every triangle t){
    Primitive primitive;
    runPrimitiveAssembly(primitive,ctx.vao,t,ctx.prg)

    runPerspectiveDivision(primitive)
    runViewportTransformation(primitive,ctx.frame)
    rasterizeTriangle(ctx.frame,primitive,ctx.prg);
  }
}

void gpu_execute(...){

}
//![gpu_execute_after_vs]




//![gpu_execute_pfo]
void runVertexAssembly(){
  computeVertexID()
  readVertexAttributes();
}

void runPrimitiveAssembly(primitive,VertexArray vao,t,vertexShader,shaderInterface){
  for(every vertex v in triangle){
    InVertex inVertex;
    runVertexAssembly(inVertex,vao,t+v);
    vertexShader(primitive.vertex,inVertex,shaderInterface);
  }
}

void rasterizeTriangle(framebuffer,primitive,fragmentShader){
  for(pixels in frame){
    if(pixels in primitive){
      InFragment inFragment;
      createFragment(inFragment,primitive,barycentrics,pixelCoord,prg);
      OutFragment outFragment;
      ShaderInterface si;
      si.uniforms = ...;
      si.textures = ...;
      
      fragmentShader(outFragment,inFragment,si);
      clampColor(outFragment,0,1);
      perFragmentOperations(framebuffer,outFragment,inFragment.gl_FragCoord.z)
    }
  }

}

void draw(GPUMemory&mem,DrawCommand const&cmd){
  for(every triangle t){
    Primitive primitive;
    runPrimitiveAssembly(primitive,vao,t,vertexShader,shaderInterface)

    runPerspectiveDivision(primitive)
    runViewportTransformation(primitive,width,height)
    rasterizeTriangle(framebuffer,primitive,fragmentShader);
  }
}

void gpu_execute(...){
}

//![gpu_execute_pfo]



//![gpu_execute]
void runVertexAssembly(){
  computeVertexID()
  readVertexAttributes();
}

void runPrimitiveAssembly(primitive,vertexArray,t,vertexShader,shaderInterface){
  for(every vertex v in triangle){
    InVertex inVertex;
    runVertexAssembly(inVertex,vertexArray,t+v);
    vertexShader(primitive.vertex,inVertex,shaderInterface);
  }
}

void rasterizeTriangle(framebuffer,primitive,fragmentShader){
  for(pixel in framebuffer){
    if(pixel in primitive){
      InFragment inFragment;
      createFragment(inFragment,primitive,barycentrics,pixelCoord,vs2fs);
      OutFragment outFragment;
      fragmentShader(outFragment,inFragment,shaderInterface);
      clampColor(outFragment,0,1);
      perFragmentOperations(framebuffer,outFragment,inFragment.gl_FragCoord.z)
    }
  }
}

void draw(mem,drawCommand,gl_DrawID){
  for(every triangle t){
    Primitive primitive;
    runPrimitiveAssembly(primitive,vertexArray,t,vertexShader,gl_DrawID)

    ClippedPrimitive clipped;
    performeClipping(clipped,primitive);

    for(all clipped triangle c in clipped){
      runPerspectiveDivision(c)
      runViewportTransformation(c,width,height)
      rasterizeTriangle(framebuffer,c,fragmentShader);
    }
  }
}

void clear(mem,clearCommad){
}

void gpu_execute(mem,commandBuffer){
  for(every command in commandBuffer){
    gl_DrawID = computeDrawID();
    if(isClearCommand)clear(mem,clearCommand)
    if(isDrawCommand )draw (mem,drawCommand,gl_DrawID)

  }
}

//![gpu_execute]


//![prepareModel]
void prepareNode(GPUMemory&mem,CommandBuffer&cb,Node const&node,Model const&model,glm::mat4 const&prubeznaMatice,...){
  if(node.mesh>=0){
    mesh = model.meshes[node.mesh];

    //tvorba kreliciho prikazu
    mesh.doubleSided ...;
    mesh.position ...;
    mesh.normal ...;
    mesh.texCoord ...;
    mesh.indexBufferID ...;
    mesh.indexOffset ...;
    mesh.indexType ...
    mesh.nofVertices ...
    createDrawCall(cb)

    //zapis dat do paměti (uniformy pro draw call)
    ZKOBINUJ(prubeznaMatice,node.modelMatrix);
    inverzni tranponovana...
    mesh.diffuseColor
    mesh.doubleSided
    writeToMemory(mem);
  }

  for(size_t i=0;i<node.children.size();++i)
    prepareNode(mem,node.children[i],model,...); rekurze
}

void prepareModel(GPUMemory&mem,CommandBuffer&cb,Model const&model){
  mem.buffers  = ...;
  mem.textures = ...;
  mem.programs = ...;

  clearCommand ...;

  glm::mat4 jednotkovaMatrice = glm::mat4(1.f);
  for(size_t i=0;i<model.roots.size();++i)
    prepareNode(mem,cb,model.roots[i],jednotkovaMatrice,...);
}
//![prepareModel]

///\endcond
