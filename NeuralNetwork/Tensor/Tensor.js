/**
 * @typedef {Object} Tensor
 * @property {TypedArray|GPUBuffer} data
 * @property {Int32Array|GPUBuffer} size
 * @property {TypedArray} type
 * @property {Number} [host_id]
 * @property {GPUDevice} [host_device] ?????
 */

import * as Buffer from '../../GPU-Connector-API/buffer.js'
import { F32_to_I32 } from './shaders/typeConversion.js';

const HOST_TYPES = {
    MAIN_HOST: { baseName: 'MAIN' },
    GPU_DEVICE_HOST: { baseName: 'GPU_DEVICE' , allowedTypes:  {
        Float32Array,
        Int32Array,
        Uint32Array,
    } },
    WEB_WORKER_HOST: { baseName: 'WEB_WORKER' },
}
const HOST_TYPES_TO_IDS = Object.fromEntries(Object.keys(HOST_TYPES).map((k, i) => [k, i]));
const HOST_IDS_TO_TYPES = Object.fromEntries(Object.keys(HOST_TYPES).map((k, i) => [i, k]));

const TypedArray = Object.getPrototypeOf(Int8Array);

const tensorHandler = {
    get(target, property, receiver) {
        if (property == 'hostName') {
            if (target.host_device) {
                return `${HOST_TYPES.GPU_DEVICE_HOST.baseName}:${target.host_id}`
            } else if (target.host_id) {
                return `${HOST_TYPES.WEB_WORKER_HOST.baseName}:${target.host_id}`
            } else {
                return HOST_TYPES.MAIN_HOST.baseName
            }
        } else if (property == 'host') {
            if (target.host_device) {
                return HOST_TYPES_TO_IDS.GPU_DEVICE_HOST
            } else if (target.host_id) {
                return HOST_TYPES_TO_IDS.WEB_WORKER_HOST
            } else {
                return HOST_TYPES_TO_IDS.MAIN_HOST
            }
        }
        return Reflect.get(target, property, receiver);
    },

    set(target, property, value, reciever) {
        if (property == 'host_id' || property == 'host_device' || property == 'type') return
        Reflect.set(target, property, value, reciever)
    }
}


function validate(data, size) {
    return data.length == size.reduce((a, c) => a * c);
}

export async function toString(t) {
    const mainTensor = await sendToMain(t)
    return `Tensor {
        size: ${mainTensor.size},
        data: ${mainTensor.data}
        type: ${t.type.name}
    }`
}


export function get(/**@type {Tensor}*/ t, ...idx) {
    const cumulativeSteps = t.size.reduce((a, c) => [...a, a.at(-1) * c], [1]);
    return t.data[idx.reduce((a, c, i) => a + c * cumulativeSteps[i])];
}

export async function sendToMain(t) {
    switch (t.host) {
        case HOST_TYPES_TO_IDS.GPU_DEVICE_HOST:
            const [copiedData, copiedSize] = await Promise.all([
                Buffer.extract(t.host_device, t.data, t.type),
                Buffer.extract(t.host_device, t.size, Uint32Array)
            ])
            return from(copiedData, copiedSize, t.type)
        case HOST_TYPES_TO_IDS.WEB_WORKER_HOST:
            // ???
            break;
        default:
            break;
    }
}

export async function sendToGPUDevice(t, device) {
    switch (t.host) {
        case HOST_TYPES_TO_IDS.MAIN_HOST:
            const [copiedData, copiedSize] = await Promise.all([
                Buffer.from(device, t.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'tensor_data'),
                Buffer.from(device, t.size, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'tensor_size'),
            ])

            return tensorProxy({ 
                data: copiedData,
                size: copiedSize,
                type: t.type,
                host_device: device,
            })
        case HOST_TYPES_TO_IDS.WEB_WORKER_HOST:
            // ???
            break;
        default:
            break;
    }
}

export function sendToWorker(t) {
    
}

function tensorProxy(tensorData) { return new Proxy(tensorData, tensorHandler) }

export function from(data, size, type) {
    if (!data) return;

    size = new Uint32Array(size || [data.length, 1]);
    type = type || Float32Array

    if (!validate(data, size)) return;

    if (!(data instanceof TypedArray)) {
        data = (type).from(data)
    }

    return tensorProxy({ data, size, type })
}

export async function clone(t) {
    let data, size;
    switch (t.host) {
        case HOST_TYPES_TO_IDS.GPU_DEVICE_HOST:
            [data, size] =  await Promise.all([
                Buffer.clone(t.host_device, t.data),
                Buffer.clone(t.host_device, t.size),
            ])
            break

        default:
            data = new (t.type)(t.data)
            size = new Uint32Array(t.size)
    }

    return tensorProxy({
        ...t,
        data,
        size,
    })
}

async function convert(t, TypedArray, mapFn) {
    console.time('convert')
    if (t.type == TypedArray) return;
       
    const host = t.host;
    const allowedTypes = HOST_TYPES[HOST_IDS_TO_TYPES[host]].allowedTypes
    if (allowedTypes && !allowedTypes[TypedArray.name]) return;

    switch (t.host) {
        case HOST_TYPES_TO_IDS.GPU_DEVICE_HOST: {

            // const mainTensor = await sendToMain(t);
            // const newData = TypedArray.from(mainTensor.data, mapFn);
            // const newTensor = from(newData, t.size);
            // return await sendToGPUDevice(newTensor, t.host_device)

            const clonedTensor = await clone(t)

            const convertShader = t.host_device.createShaderModule({
                code: F32_to_I32
            })

            const bindGroupLayout = t.host_device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                ]
            });

            const pipelineLayout = t.host_device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            });

            const computePipeline = t.host_device.createComputePipeline({
                layout: pipelineLayout,
                compute: {
                    module: convertShader,
                    entryPoint: 'convert',
                },
            });

            const bindGroup = t.host_device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: t.data} },
                    { binding: 1, resource: { buffer: clonedTensor.data} }
                ]
            });

            const commandEncoder = t.host_device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
        
            // Dispatch Compute Workgroups
            passEncoder.dispatchWorkgroups(t.data.size / (256 * 4));
            
            passEncoder.end();
        
            
            const commands = commandEncoder.finish();
            t.host_device.queue.submit([commands]);
            await t.host_device.queue.onSubmittedWorkDone();

            console.timeEnd('convert')
            return tensorProxy({ ...clonedTensor, type: TypedArray })
        } default: {
            const newData = TypedArray.from(t.data, mapFn);
            console.timeEnd('convert')
            return from(newData, t.size);
        }
    }
}

export async function extractData(t) {
    if (t.data instanceof TypedArray) return t.data;
    return await Buffer.extract(t.host_device, t.data, t.type)
}

export async function extractSize(t) {
    if (t.size instanceof TypedArray) return t.size;
    return await Buffer.extract(t.host_device, t.size, Uint32Array)
}

export async function transform(t, mapFn) {
    switch (t.host) {
        case HOST_TYPES_TO_IDS.GPU_DEVICE_HOST: {
            const [originalData, size] = await Promise.all([
                Buffer.extract(t.host_device, t.data, t.type),
                Buffer.extract(t.host_device, t.size, Uint32Array)
            ]);

            const newData = (t.type).from(originalData, mapFn);
            const newTensor = from(newData, size)

            return sendToGPUDevice(newTensor, t.device);
        } default: {
            const newData = (t.type).from(t.data, mapFn);
            return from(newData, t.size);
        }
    }
}

export async function round(t) {
    return await transform(t, Math.round)
}

export async function trunc(t) {
    return await transform(t, Math.trunc)
}

export async function floor(t) {
    return await transform(t, Math.floor)
}

export async function ceil(t) {
    return await transform(t, Math.ceil)
}

// Int
export async function toInt64(t) {
    return await convert(t, BigInt64Array);
}

export async function toInt32(t) {
    return await convert(t, Int32Array);
}

export async function toInt16(t) {
    return await convert(t, Int16Array);
}

export async function toInt8(t) {
    return await convert(t, Int8Array);
}

export const toInt = toInt32;

// Uint
export async function toUint64(t) {
    return await convert(t, BigUint64Array);
}

export async function toUint32(t) {
    return await convert(t, Uint32Array);
}

export async function toUint16(t) {
    return await convert(t, Uint16Array);
}

export async function toUint8(t) {
    return await convert(t, Uint8Array);
}

export const toUint = toUint32;

// Float
export async function toFloat64(t) {
    return await convert(t, Float64Array);
}

export async function toFloat32(t) {
    return await convert(t, Float32Array);
}

export const toFloat = toFloat32;
