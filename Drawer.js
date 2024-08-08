export default class Drawer {
    canvas = document.querySelector('canvas') || document.createElement('canvas');
    context;
    cursor = {x: -1, y: -1, z: -1};
    updateRender = true;
    hardUpdate = true;
    gpuData = {
        ratio: 1,
        DEVICE: null,
        buffers: {},
        shaders: {},
        renderPassDescriptor: {
            colorAttachments: [
              {
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                loadOp: "clear",
                storeOp: "store",
                view: null,
              },
            ],
            depthStencilAttachment: {
                view: null,
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
              },
        }
    }

    constructor() {
        this.#setUpCanvasResizeListener()
    }

    async init(contextType) {
        this.context = this.canvas.getContext(contextType);
        if (!this.context) throw new Error('invalid context');

        if (contextType === 'webgpu') await this.#initGPU();
    }

    async #initGPU() {
        if (!navigator.gpu) throw Error("WebGPU not supported.");

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw Error("Couldn't request WebGPU ADAPTER.");

        const device = await adapter.requestDevice();
        this.gpuData.DEVICE = device;

        this.context.configure({
            device: device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: "premultiplied",
            usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    createShader(name, code) {
        this.setShader(name, this.gpuData.DEVICE.createShaderModule({ code }));
    }

    getShader(name) {
        return this.gpuData.shaders[name];
    }

    setShader(name, shader) {
        this.gpuData.shaders[name] = shader;
    }

    createBuffer(name, size, usage) {
        this.setBuffer(name, this.gpuData.DEVICE.createBuffer({ size, usage }));
    }

    writeBuffer(name, bufferOffset, data, dataOffset, dataSize) {
        this.gpuData.DEVICE.queue.writeBuffer(this.getBuffer(name), bufferOffset, data, dataOffset, dataSize);
    }

    writeBuffer1to1(name, data) {
        this.writeBuffer(name, 0, data, 0, data.length);
    }

    getBuffer(name) {
        return this.gpuData.buffers[name];
    }

    setBuffer(name, buffer) {
        this.gpuData.buffers[name] = buffer;
    }


    #resizeCanvas() {
        const DEVICE_PIXEL_RATIO = window.devicePixelRatio;
        this.canvas.height = this.canvas.offsetHeight * DEVICE_PIXEL_RATIO;
        this.canvas.width = this.canvas.offsetWidth * DEVICE_PIXEL_RATIO;
        return [this.canvas.width, this.canvas.height];
    }

    #setUpCanvasResizeListener() {
        const work = () => {
            const [xLen, yLen] = this.#resizeCanvas();
            this.gpuData.ratio = xLen / yLen;
        }
        const wrappedResize = () => {
            document.removeEventListener("DOMContentLoaded", wrappedResize)
            work();
        }
        window.addEventListener("resize", work)
        document.addEventListener("DOMContentLoaded", wrappedResize);
    }

    setUpCursorListener() {
        this.canvas.addEventListener("mousemove", (e) => {
            if (e.offsetX <= this.canvas.offsetWidth && e.offsetX >= 0) {
                this.cursor.y = e.offsetX / this.canvas.offsetWidth;
                this.updateRender = true;
            }
            if (e.offsetY <= this.canvas.offsetHeight && e.offsetY >= 0) {
                this.cursor.x = e.offsetY / this.canvas.offsetHeight;
                this.updateRender = true;
            }
        })
    }

    setUpKeyListener(keyPredicate, work) {
        document.addEventListener('keydown', (e) => {
            if (keyPredicate(e.key)) work(e.key);
        });
    }


}