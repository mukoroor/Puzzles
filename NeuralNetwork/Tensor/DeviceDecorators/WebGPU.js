import { Buffer } from "../../../GPU-Connector-API/index.js"
import { getConverisonShader, TYPE_CONVERSION_ENTRY_POINT } from "../shaders/typeConversion.js"

const USAGE_MAP = {
    'tensor_data' : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    'tensor_size' : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
}

const DEVICE_MAP = {

}

function vec4_aligned_size(size) { 
    return size.map(e => e % 4 ? (e + 3) & ~3: e)
}

function dataToVec4Aligned(data, size) {
    const aligned_size = vec4_aligned_size(size)

    const col_diff = new Array(aligned_size[1] - size[1]).fill(0);

    const alignedData = new Array(aligned_size[0])

    for (let i = 0; i < alignedData.length; i++) {
        if (i >= size[0]) {
            alignedData[i] = new Array(aligned_size[1]).fill(0)
        } else {
            alignedData[i] = [...data.slice(i * size[1], (i + 1) * size[1]), ...col_diff]
        }
    }
    return new data.constructor(alignedData.flat())
} 

function vec4unaligned(data, size) {
    const aligned_size = vec4_aligned_size(size)

    const unalignedData = new Array(size[0])

    for (let i = 0; i < unalignedData.length; i++) {
        const base = i * aligned_size[1];
        unalignedData[i] = [...data.slice(base, base + size[1])]
    }

    return new data.constructor(unalignedData.flat())
} 

export function allocate(device, size, usage, label) {
    const total_allocation_size = vec4_aligned_size(size).reduce((a, c) => c ? a * c : a)

    return Buffer.allocate(device, total_allocation_size, USAGE_MAP[usage], label)
}

export function write(device, data, size, destination, dataOffset=0, destOffset=0, dataSize) {
    const aligned_data = dataToVec4Aligned(data, size);

    Buffer.write(device, destination, aligned_data, dataOffset, destOffset, dataSize)
}

export function initialize(device, data, size, usage, label) {
    const aligned_data = dataToVec4Aligned(data, size);

    return Buffer.from(device, aligned_data, USAGE_MAP[usage], label)
}

export async function clone(device, source, usage, sourceOffset=0, size) {
    return await Buffer.clone(device, source, USAGE_MAP[usage], sourceOffset, size)
}

export async function registerDevice(device, id) {
    DEVICE_MAP[id] = device;
}

export async function sendToMainThread(device, data, size, type) {
    const mappedData = await Buffer.extract(device, data, type)

    return vec4unaligned(mappedData, size)
}

export async function toString(device, data, type) {
    return sendToMainThread(device, data, type);
}

export async function convert(device, data, type) {
    const clonedData = await clone(device, data, USAGE_MAP.tensor_data)

    const convertShader = device.createShaderModule({
        code: getConverisonShader(data.type, type)
    })

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const computePipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: convertShader,
            entryPoint: TYPE_CONVERSION_ENTRY_POINT,
        },
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: data} },
            { binding: 1, resource: { buffer: clonedData} }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch Compute Workgroups
    passEncoder.dispatchWorkgroups(data.size / (256 * 4));
    
    passEncoder.end();

    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    await device.queue.onSubmittedWorkDone();

    return clonedData
}

export async function transform(device, data, mapFn) {}

function schedule() {}


// async function convert(t, TypedArray, mapFn) {
//     console.time('convert')
//     if (t.type == TypedArrady) return;
       
//     const host = t.host;
//     const allowedTypes = HOST_TYPES[HOST_IDS_TO_TYPES[host]].allowedTypes
//     if (allowedTypes && !allowedTypes[TypedArray.name]) return;

//     switch (t.host) {
//         case HOST_TYPES_TO_IDS.GPU_DEVICE_HOST: {

//             // const mainTensor = await sendToMain(t);
//             // const newData = TypedArray.from(mainTensor.data, mapFn);
//             // const newTensor = from(newData, t.size);
//             // return await sendToGPUDevice(newTensor, device)

            
//         } default: {
//             const newData = TypedArray.from(t.data, mapFn);
//             console.timeEnd('convert')
//             return from(newData, t.size);
//         }
//     }
// }