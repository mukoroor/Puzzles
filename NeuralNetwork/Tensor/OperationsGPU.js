import * as Tensor from './Tensor.js'
import { from as buffer_from } from '../../GPU-Connector-API/buffer.js';
import { SHADER, SHADER_ENTRY_POINT, TILE_BLOCK_DIM, TILE_SIZE } from './shaders/MatrixMultiplication.js';

function buildOperationPipeline(device, shaderModule, shaderEntryPoint, buffers) {
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const computePipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: shaderEntryPoint,
        },
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: buffers.map((e, i) => Object({ binding: i, resource: { buffer: e} }))
    });

    return [computePipeline, bindGroup]
}




export default async function matrixMultiplication(a, b, output) {
  console.time('mult')
  const device = a.device

  // Load WGSL Shader
  const shaderModule = device.createShaderModule({
      code: SHADER,
  });

  if (a.type != b.type || (output && b.type != output.type)) throw new Error('type mismatch')
  if (device != b.device || (output && device != output.device)) throw new Error('tensor on different devices')

  let aDims = a.size, bDims = b.size, outDims;
  if (output) {
      outDims = output.size;
      if (outDims[0] != aDims[0] || outDims[1] != bDims[1]) throw new Error('Incorrect Dimensions')
  } else {
    outDims = [aDims[0], bDims[1]]

    output = Tensor.empty(outDims, a.type, device)
  }
  
  const dims_arr = [...aDims, ...bDims, ...outDims].map(e => e % 4 ? (e + 3) & ~3: e)

  const allDims = await buffer_from(device, new Uint32Array(dims_arr) , GPUBufferUsage.UNIFORM)
  const [computePipeline, bindGroup] = buildOperationPipeline(device, shaderModule, SHADER_ENTRY_POINT, 
    [
        a.data,
        b.data,
        output.data,
        allDims
    ])

  // Create Command Encoder & Compute Pass
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);

  // Dispatch Compute Workgroups
  const TILE_BLOCK_WIDTH = TILE_BLOCK_DIM * TILE_SIZE;
  const workgroupCountX = Math.ceil(outDims[0] / TILE_BLOCK_WIDTH);
  const workgroupCountY = Math.ceil(outDims[1] / TILE_BLOCK_WIDTH);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
 
  passEncoder.end();

  
  const commands = commandEncoder.finish();
  device.queue.submit([commands]);
  await device.queue.onSubmittedWorkDone();
  console.timeEnd('mult')

  return output
}