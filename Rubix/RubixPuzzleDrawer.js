import Drawer from "../Drawer.js";
import { ROUNDED_SQUARE_INDICES, ROUNDED_SQUARE_VERTICES } from "./Meshes.js";
import RubixPuzzle from "./RubixPuzzle.js";
import { frag_vert_shader } from "./RubixShader.js";



export default class RubixPuzzleDrawer extends Drawer {
    static MESH_DATA = Object.freeze({
        TRIANGLES: new Float32Array(ROUNDED_SQUARE_INDICES.flat().reduce((arr, vertexIndex) => arr.concat(ROUNDED_SQUARE_VERTICES[vertexIndex]), []).flat()),
        TRIANGLE_ENUMERATION: new Float32Array(ROUNDED_SQUARE_INDICES.reduce((arr, _, i) => arr.concat([i, i, i]), []))
    });
    static PIECE_FACE_DELTAS = Object.freeze({
        0: [0, 1, 0],
        1: [0, 0, -1],
        2: [1, 0, 0],
    })

    puzzle = new RubixPuzzle(2);
    
    constructor() {
        super();
        this.puzzle.faces[0].rotate('cw', 1);
        this.puzzle.faces[1].rotate('cw', 1);
        this.setUpCursorListener();
        this.setUpKeyListener('ArrowUp', () => this.#incrementPuzzleSize(1));
        this.setUpKeyListener('ArrowDown', () => this.#incrementPuzzleSize(-1));
    }

    async draw() {
        const positions = this.#calculatePieceCoordinates();
        await this.setUpBuffers(positions);
        
        const frame = async() => {
            if (this.updateRender) {
                this.updateRender = false;
                this.render(this.createRenderPipeline(), positions.length)
            }
            window.requestAnimationFrame(frame);
        }
        window.requestAnimationFrame(frame);
    }

    async setUpBuffers(piecePositions) {
        if (!this.gpuData.DEVICE) await this.init("webgpu");
        console.log(piecePositions)

        // const VERTICES = new Float32Array(ROUNDED_SQUARE_VERTICES.flat())
        // const INDICES = new Uint32Array(ROUNDED_SQUARE_INDICES.flat())

        const PIECE_POSITIONS = new Float32Array(piecePositions.map(e => e[0]).flat());
        const PIECE_COLORS = new Uint32Array(piecePositions.map(e => e[1]).flat());

        // this.createBuffer("vertex_buffer", VERTICES.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        // this.createBuffer("index_buffer", INDICES.byteLength, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.createBuffer("triangles_buffer", RubixPuzzleDrawer.MESH_DATA.TRIANGLES.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.createBuffer("triangle_enum", RubixPuzzleDrawer.MESH_DATA.TRIANGLE_ENUMERATION.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.createBuffer("piece_positions", PIECE_POSITIONS.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.createBuffer("piece_colors", PIECE_COLORS.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

        // this.writeBuffer1to1("vertex_buffer", VERTICES);
        // this.writeBuffer1to1("index_buffer", INDICES);
        this.writeBuffer1to1("triangles_buffer", RubixPuzzleDrawer.MESH_DATA.TRIANGLES);
        this.writeBuffer1to1("triangle_enum", RubixPuzzleDrawer.MESH_DATA.TRIANGLE_ENUMERATION);
        this.writeBuffer1to1("piece_positions", PIECE_POSITIONS);
        this.writeBuffer1to1("piece_colors", PIECE_COLORS);


        this.createShader("frag_vert", frag_vert_shader);
        
    }

    createRenderPipeline() {
        return this.gpuData.DEVICE.createRenderPipeline({
            layout: 'auto',
            vertex: {
                constants: { ratio: this.gpuData.ratio, scale: this.#calculateScale(), origin: this.#calculatePuzzleCenter(), ...this.cursor },
                module: this.getShader("frag_vert"),
                buffers: [
                    {
                        arrayStride: 12,
                        stepMode: 'vertex',
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x3',
                            },
                        ],
                    },

                    {
                        arrayStride: 4,
                        stepMode: 'vertex',
                        attributes: [
                            {
                                shaderLocation: 1,
                                offset: 0,
                                format: 'float32',
                            },
                        ],
                    },

                    {
                        arrayStride: 12,
                        stepMode: 'instance',
                        attributes: [
                            {
                                shaderLocation: 2,
                                offset: 0,
                                format: 'float32x3',
                            },
                        ],
                    },
                    {
                        arrayStride: 24,
                        stepMode: 'instance',
                        attributes: [
                            {
                                shaderLocation: 3,
                                offset: 0,
                                format: 'uint32x3',
                            },
                            {
                                shaderLocation: 4,
                                offset: 12,
                                format: 'uint32x3',
                            },
                        ],
                    }
                ],
            },
            fragment: {
                // constants: { ratio: this.gpuData.ratio },
                module: this.getShader("frag_vert"),
                targets: [
                    { format: navigator.gpu.getPreferredCanvasFormat() }
                ],
            },
            primitive: {
                topology: 'triangle-list',
                frontFace: 'cw',
                cullMode: 'back',
            },

            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
        });
    }

    render(pipeline) {
        let commandEncoder = this.gpuData.DEVICE.createCommandEncoder();

        const depthTexture = this.gpuData.DEVICE.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.gpuData.renderPassDescriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();
        this.gpuData.renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();

        let passEncoder = commandEncoder.beginRenderPass(this.gpuData.renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, this.getBuffer("triangles_buffer"));
        passEncoder.setVertexBuffer(1, this.getBuffer("triangle_enum"));
        passEncoder.setVertexBuffer(2, this.getBuffer("piece_positions"));
        passEncoder.setVertexBuffer(3, this.getBuffer("piece_colors"));
        // passEncoder.setIndexBuffer(this.getBuffer("index_buffer"), "uint32");
        // passEncoder.drawIndexed(INDICES.length, 8);
        passEncoder.draw(RubixPuzzleDrawer.MESH_DATA.TRIANGLES.length / 3, this.puzzle.pieceCount);
        passEncoder.end();
        this.gpuData.DEVICE.queue.submit([commandEncoder.finish()]);
    }

    #calculatePieceCoordinates() {
        const length = this.puzzle.length;

        if (length === 1) return [[[0, 0, 0], [7, 0, 0, 0, 0, 0]]];

        const pieceMap = {};
        const queue = [this.puzzle.faces[5].corners[3]];

        while(queue.length) {
            const currPiece = queue.shift();
            if (!pieceMap[currPiece.id]) pieceMap[currPiece.id] = [[0, 0, 0]];
            const coloredFaces = Array(6).fill(6);
            for (const [faceIndex, face] of currPiece.faceData.entries()) {
                if (typeof face === 'number') coloredFaces[faceIndex] = face;
                if (face?.id === undefined || pieceMap[face.id]) continue;
                const delta = this.#getPieceFaceDelta(faceIndex);
                pieceMap[face.id] = [pieceMap[currPiece.id][0].map((e, i) => e + delta[i])];
                queue.push(face);
            }
            pieceMap[currPiece.id].push(coloredFaces)
        }
        console.log(pieceMap)
        return Object.values(pieceMap);
    }

    #getPieceFaceDelta(faceIndex) {
        return RubixPuzzleDrawer.PIECE_FACE_DELTAS[faceIndex < 3 ? faceIndex : 5 - faceIndex].map(e => faceIndex < 3 ? e: -e);
    }

    #calculateScale() {
        return Math.sqrt(0.5) / this.puzzle.length;
    }

    #calculatePuzzleCenter() {
        return (this.puzzle.length - 1) / 2;
    }

    #incrementPuzzleSize(delta) {
        const newLength = this.puzzle.length + delta;
        if (newLength < 1 || newLength > 10) return;
        this.puzzle = new RubixPuzzle(newLength);
        this.updateRender = true;
    }
}