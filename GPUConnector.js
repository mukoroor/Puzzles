export default class GPUConnector {
    gpuData = {
        DEVICE: null,
        buffers: {},
        bindGroups: [],
        bindGroupLayouts: [],
        shaders: {},
        pipelines: {},
    };

    async initGPU(descriptor = {}) {
        if (!navigator.gpu) throw Error("WebGPU not supported.");

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw Error(`Couldn't request WebGPU ADAPTER.`);

        const device = await adapter.requestDevice(descriptor);
        this.device = device;
    }

    addBindGroup(bindGroup) {
        this.gpuData.bindGroups.push(bindGroup);
    }

    setPipeline(name, pipeline) {
        this.gpuData.pipelines[name] = pipeline;
    }

    getPipeline(name) {
        return this.gpuData.pipelines[name];
    }

    createShader(name, code) {
        this.setShader(name, this.device.createShaderModule({ code }));
    }

    getShader(name) {
        return this.gpuData.shaders[name];
    }

    setShader(name, shader) {
        this.gpuData.shaders[name] = shader;
    }

    createBuffer(name, size, usage) {
        if (this.device.limits.maxStorageBufferBindingSize < size)
            throw new Error("Buffer to Large");
        this.setBuffer(name, this.device.createBuffer({ label: name, size, usage }));
    }

    writeBuffer(name, bufferOffset, data, dataOffset, dataSize) {
        this.device.queue.writeBuffer(
            this.getBuffer(name),
            bufferOffset,
            data,
            dataOffset,
            dataSize
        );
    }

    writeBuffer1to1(name, data) {
        this.writeBuffer(name, 0, data, 0, data.length);
    }

    copyBuffer(name, encoder = this.device.createCommandEncoder(), offset=0, size) {
        const sourceBuff = this.getBuffer(name);
        size = size || sourceBuff.size;
        const copyName = `${name}_copy`;
        this.createBuffer(
            copyName,
            size,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );

        encoder.copyBufferToBuffer(
            sourceBuff,
            offset,
            this.getBuffer(copyName),
            0,
            size
        );
    }

    async mapBufferToCPU(name, OutTypedArrConstructor, offset, size) {
        const buff = this.getBuffer(name);
        await buff.mapAsync(GPUMapMode.READ, offset, size);
        const res = new OutTypedArrConstructor(buff.getMappedRange(offset, size));
        return res;
    }

    getBuffer(name) {
        return this.gpuData.buffers[name];
    }

    setBuffer(name, buffer) {
        this.gpuData.buffers[name] = buffer;
    }

    setTexture(name, texture) {
        this.gpuData.textures[name] = texture;
    }

    getTexture(name) {
        return this.gpuData.textures[name];
    }

    set device(newDevice) {
        this.gpuData.DEVICE = newDevice;
    }

    get device() {
        return this.gpuData.DEVICE;
    }

    static waitForAnimationFrame(work) {
        return new Promise((resolve, reject) => {
            // Request the next animation frame and resolve the promise when it is called
            const refreshId = requestAnimationFrame(async (timeStamp) => {
                try {
                    const result = await work(timeStamp);
                    resolve(result || refreshId);
                } catch (e) {
                    console.log(e);
                    reject();
                }
            });
        });
    }
}
