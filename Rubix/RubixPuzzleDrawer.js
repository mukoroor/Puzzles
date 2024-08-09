import Drawer from "../Drawer.js";
import { ROUNDED_SQUARE_INDICES, ROUNDED_SQUARE_VERTICES } from "./Meshes.js";
import RubixPiece from "./RubixPiece.js";
import RubixPuzzle from "./RubixPuzzle.js";
import { frag_vert_shader } from "./RubixShader.js";

export default class RubixPuzzleDrawer extends Drawer {
  static PIECE_FACE_DELTAS = Object.freeze({
    0: [0, 1, 0],
    1: [0, 0, -1],
    2: [1, 0, 0],
  });
  movementData = Array(2);
  puzzle = new RubixPuzzle(3);

  constructor() {
    super();
    this.setUpCursorListener();
    this.setUpKeyListener(
      (type) => type === "ArrowUp",
      () => this.#incrementPuzzleSize(1)
    );
    this.setUpKeyListener(
      (type) => type === "ArrowDown",
      () => this.#incrementPuzzleSize(-1)
    );
    this.setUpKeyListener(
      (type) => type === "r",
      () => {
        this.puzzle = new RubixPuzzle(this.puzzle.length);
        this.updateFlag = 2;
      }
    );
    this.setUpKeyListener(
      (type) => type === "s",
      () => {
        this.puzzle.scramble();
        this.updateFlag = 1;
      }
    );
    this.setUpKeyListener(
      (type) => type === "e",
      () => {
        this.puzzle.scramble(10);
        this.updateFlag = 1;
      }
    );
    this.setUpKeyListener(
      (type) => type === "c",
      () => this.#rotate("cw")
    );
    this.setUpKeyListener(
      (type) => type === "w",
      () => this.#rotate("ccw")
    );
    this.setUpKeyListener(
      (type) => type === "+",
      () => this.#changeDepth(1)
    );
    this.setUpKeyListener(
      (type) => type === "-",
      () => this.#changeDepth(-1)
    );
    this.setUpKeyListener(
      (type) => !isNaN(type),
      (faceId) => this.#changeFace(+faceId)
    );
  }

  async init() {
    await super.init("webgpu");

    this.#createMeshBuffers();
    this.createShader("frag_vert", frag_vert_shader);
    this.setRenderPipeline("main", this.#createMainRenderPipeline());
    this.updateFlag = 2;

    this.gpuData.renderPassDescriptor.depthStencilAttachment.view =
      this.gpuData.DEVICE.createTexture({
        size: [this.canvas.width, this.canvas.height],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      }).createView();
  }

  async draw() {
    if (!this.gpuData.DEVICE) await this.init();

    let data, colors;
    const frame = async () => {
      switch (this.updateFlag) {
        case Number.MAX_SAFE_INTEGER:
          this.setRenderPipeline("main", this.#createMainRenderPipeline());
          this.render();
          break;
        case 2:
          this.#destroyBuffers();
          data = this.#generatePieceVisitSequenceAndPositions();
          this.#createAndFillPiecePositionBuffer(data.positions);
          if (this.movementData[1] > this.puzzle.length - 1)
            this.movementData[0] = this.puzzle.length - 1;
        case 1:
          colors = this.#generatePieceColors(data.sequence);
          this.#createAndFillPieceColorBuffer(colors);
        case 0:
          this.render();
          break;
        default:
          break;
      }
      this.updateFlag = -1;
      window.requestAnimationFrame(frame);
    };
    window.requestAnimationFrame(frame);
  }

  #createMeshBuffers() {
    const TRIANGLES = new Float32Array(
      ROUNDED_SQUARE_INDICES.flat()
        .reduce(
          (arr, vertexIndex) =>
            arr.concat(ROUNDED_SQUARE_VERTICES[vertexIndex]),
          []
        )
        .flat()
    );
    const TRIANGLE_ENUMERATION = new Float32Array(
      ROUNDED_SQUARE_INDICES.reduce((arr, _, i) => arr.concat([i, i, i]), [])
    );
    this.createBuffer(
      "triangles_buffer",
      TRIANGLES.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "triangle_enum",
      TRIANGLE_ENUMERATION.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.writeBuffer1to1("triangles_buffer", TRIANGLES);
    this.writeBuffer1to1("triangle_enum", TRIANGLE_ENUMERATION);
  }

  #destroyBuffers() {
    this.getBuffer("piece_positions")?.destroy();
    this.getBuffer("piece_colors")?.destroy();
  }

  #createAndFillPiecePositionBuffer(positions) {
    const PIECE_POSITIONS = new Float32Array(positions.flat());

    this.createBuffer(
      "piece_positions",
      PIECE_POSITIONS.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    this.writeBuffer1to1("piece_positions", PIECE_POSITIONS);
  }

  #createAndFillPieceColorBuffer(colors) {
    const PIECE_COLORS = new Uint32Array(colors.flat());
    this.createBuffer(
      "piece_colors",
      PIECE_COLORS.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.writeBuffer1to1("piece_colors", PIECE_COLORS);
  }

  #createMainRenderPipeline() {
    return this.gpuData.DEVICE.createRenderPipeline({
      layout: "auto",
      vertex: {
        constants: {
          ratio: this.gpuData.ratio,
          scale: this.#calculateScale(),
          origin: this.#calculatePuzzleCenter(),
          ...this.cursor,
        },
        module: this.getShader("frag_vert"),
        buffers: [
          {
            arrayStride: 12,
            stepMode: "vertex",
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3",
              },
            ],
          },

          {
            arrayStride: 4,
            stepMode: "vertex",
            attributes: [
              {
                shaderLocation: 1,
                offset: 0,
                format: "float32",
              },
            ],
          },

          {
            arrayStride: 12,
            stepMode: "instance",
            attributes: [
              {
                shaderLocation: 2,
                offset: 0,
                format: "float32x3",
              },
            ],
          },
          {
            arrayStride: 24,
            stepMode: "instance",
            attributes: [
              {
                shaderLocation: 3,
                offset: 0,
                format: "uint32x3",
              },
              {
                shaderLocation: 4,
                offset: 12,
                format: "uint32x3",
              },
            ],
          },
        ],
      },
      fragment: {
        // constants: { ratio: this.gpuData.ratio },
        module: this.getShader("frag_vert"),
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: {
        topology: "triangle-list",
        frontFace: "cw",
        cullMode: "back",
      },

      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth32float",
      },
    });
  }

  render() {
    this.gpuData.renderPassDescriptor.colorAttachments[0].view = this.context
      .getCurrentTexture()
      .createView();

    let commandEncoder = this.gpuData.DEVICE.createCommandEncoder();
    let passEncoder = commandEncoder.beginRenderPass(
      this.gpuData.renderPassDescriptor
    );

    passEncoder.setPipeline(this.getRenderPipeline("main"));

    passEncoder.setVertexBuffer(0, this.getBuffer("triangles_buffer"));
    passEncoder.setVertexBuffer(1, this.getBuffer("triangle_enum"));
    passEncoder.setVertexBuffer(2, this.getBuffer("piece_positions"));
    passEncoder.setVertexBuffer(3, this.getBuffer("piece_colors"));

    passEncoder.draw(
      this.getBuffer("triangles_buffer").size /
        (3 * Float32Array.BYTES_PER_ELEMENT),
      this.puzzle.pieceCount
    );
    passEncoder.end();
    this.gpuData.DEVICE.queue.submit([commandEncoder.finish()]);
  }

  #generatePieceColors(sequence) {
    if (sequence.length === 1) return [7, 0, 0, 0, 0, 0];
    return sequence.map((e) => {
      const color = [];
      for (const face of e.faceData) {
        color.push(face === undefined || face instanceof RubixPiece ? 6 : face);
      }
      return color;
    });
  }

  #generatePieceVisitSequenceAndPositions() {
    const length = this.puzzle.length;

    const pieceMap = new Map();
    const removalQueue = [
      length === 1
        ? this.puzzle.faces[5].center
        : this.puzzle.faces[5].corners[3],
    ];
    const sequence = [];
    const positions = [];

    while (removalQueue.length) {
      const currPiece = removalQueue.shift();
      sequence.push(currPiece);

      if (!pieceMap.has(currPiece.id)) {
        const index = positions.push([0, 0, 0]) - 1;
        pieceMap.set(currPiece.id, index);
      }

      for (const [faceIndex, face] of currPiece.faceData.entries()) {
        if (face?.id === undefined || pieceMap.has(face.id)) continue;
        const delta = this.#getPieceFaceDelta(faceIndex);
        const position = positions[pieceMap.get(currPiece.id)].map(
          (e, i) => e + delta[i]
        );
        pieceMap.set(face.id, positions.push(position) - 1);
        removalQueue.push(face);
      }
    }

    return { sequence, positions };
  }

  #getPieceFaceDelta(faceIndex) {
    return RubixPuzzleDrawer.PIECE_FACE_DELTAS[
      faceIndex < 3 ? faceIndex : 5 - faceIndex
    ].map((e) => (faceIndex < 3 ? e : -e));
  }

  #calculateScale() {
    return Math.sqrt(0.5) / this.puzzle.length;
  }

  #calculatePuzzleCenter() {
    return (this.puzzle.length - 1) / 2;
  }

  #incrementPuzzleSize(delta) {
    const newLength = this.puzzle.length + delta;
    if (newLength < 1 || newLength > 25) return;
    this.puzzle = new RubixPuzzle(newLength);
    this.updateFlag = 2;
    console.log(
      `%cNEW RUBIX SIZE ${newLength}`,
      "font-weight: bold; color: orange; font-size: 1.25em;"
    );
  }

  #changeFace(faceId) {
    if (faceId >= 0 && faceId < 6) {
      this.movementData[0] = faceId;
    }
  }

  #changeDepth(delta) {
    let newDepth = (this.movementData[1] || 0) + delta;
    if (newDepth < 0 || newDepth > this.puzzle.length - 1) return;
    this.movementData[1] = newDepth;
  }

  #rotate(direction) {
    if (this.movementData[0] === undefined) return;
    this.puzzle.rotate(
      this.movementData[0],
      1,
      direction,
      this.movementData[1] || 0
    );
    this.updateFlag = 1;
  }
}
