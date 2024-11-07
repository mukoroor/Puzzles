import Drawer from "../Drawer.js";
import { ROUNDED_SQUARE_INDICES, ROUNDED_SQUARE_VERTICES } from "./Meshes.js";
import RubixPuzzle from "./RubixPuzzle.js";
import { frag_vert_shader } from "./RubixShader.js";
import { frag_vert_aid_shader } from "./AidShader.js";
import { AID_MESH } from "./Meshes.js";

export default class RubixPuzzleDrawer extends Drawer {
  static UPDATE_FLAGS = Object.freeze({
    COLORS_CHANGE: 3,
    ROTATION: 2,
    INTERPOLATION: 1,
  });
  static PIECE_FACE_DELTAS = Object.freeze({
    0: [0, 1, 0],
    1: [0, 0, -1],
    2: [1, 0, 0],
  });
  colors = [
    [1, 1, 1, 1],
    [0, 0.5, 0, 1],
    [1, 0.55, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
  ];
  movementData = [0, 0, "", 1];
  rotationAngle = 0;
  rotationSpeed = 10;
  showFaceAid = true;
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
      (type) => type === "r" && this.puzzle.length > 1,
      () => {
        this.puzzle = new RubixPuzzle(this.puzzle.length);
        this.updateFlag = Drawer.UPDATE_FLAGS.RESET;
      }
    );
    this.setUpKeyListener(
      (type) => type === "s" && this.puzzle.length > 1,
      () => {
        this.puzzle.scramble();
        this.updateFlag = RubixPuzzleDrawer.UPDATE_FLAGS.ROTATION;
      }
    );
    this.setUpKeyListener(
      (type) => type === "e" && this.puzzle.length > 1,
      () => {
        this.puzzle.crossPattern();
        this.updateFlag = RubixPuzzleDrawer.UPDATE_FLAGS.ROTATION;
      }
    );
    this.setUpKeyListener(
      (type) => type === "c" && this.puzzle.length > 1,
      () => {
        this.movementData[2] = "cw";
        this.#startInterpolation();
      }
    );
    this.setUpKeyListener(
      (type) => type === "w" && this.puzzle.length > 1,
      () => {
        this.movementData[2] = "ccw";
        this.#startInterpolation();
      }
    );
    this.setUpKeyListener(
      (type) => type === "u" && this.puzzle.length > 1,
      () => this.#unscramble()
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
    this.setUpKeyListener(
      (type) => type === ">",
      () => (this.movementData[3] = this.movementData[3] == 3 ?  1 : (this.movementData[3] + 1) % 4)
    );
    this.setUpKeyListener(
      (type) => type === "?",
      () => {
        this.colors = generateHSLColors(6).map((e) => hslToRgba(...e));
        this.updateFlag = RubixPuzzleDrawer.UPDATE_FLAGS.COLORS_CHANGE;
      }
    );
    this.setUpKeyListener(
      (type) => type === "a",
      () => {
        this.showFaceAid = !this.showFaceAid;
        this.updateFlag = Drawer.UPDATE_FLAGS.FORCE_RENDER;
      }
    );
  }

  async init() {
    await super.init("webgpu");

    this.createShader("frag_vert", frag_vert_shader);
    this.createShader("frag_vert_aid", frag_vert_aid_shader);
    this.createSampler("sampler_1");
    await Promise.all(
      Array.from({ length: this.puzzle.faces.length }, (_, i) => {
        return this.createTextureFromImage(
          `./Rubix/face_textures/${i}.png`,
          `face_image_${i}`
        );
      })
    );

    this.gpuData.renderPassDescriptor.depthStencilAttachment.view = this.device
      .createTexture({
        size: [this.canvas.width, this.canvas.height],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      .createView();
  }

  async draw() {
    if (!this.device) await this.init();

    let data,
      colors,
      rotationData = [];
    const frame = async () => {
      switch (this.updateFlag) {
        case RubixPuzzleDrawer.UPDATE_FLAGS.COLORS_CHANGE:
          this.#updateColorsBuffer();
          this.render();
          break;
        case Drawer.UPDATE_FLAGS.CURSOR_UPDATE:
          this.#updateRenderStateBuffer();
          this.render();
          break;
        case Drawer.UPDATE_FLAGS.RESET:
          this.#destroyBuffers();
          this.#createMeshBuffers();
          this.#createBindBuffers();

          this.#updateRenderStateBuffer();
          this.#updateColorsBuffer();

          this.gpuData.bindGroups = [];
          this.setRenderPipeline("main", this.#createMainRenderPipeline());
          this.setRenderPipeline("aid", this.#createFaceAidRenderPipeline());

          data = this.#getPiecesVisitSequenceAndPositions();
          this.#updatePiecePositionsBuffer(data.positions);
          if (this.movementData[1] > this.puzzle.length - 1)
            this.movementData[0] = this.puzzle.length - 1;
        case RubixPuzzleDrawer.UPDATE_FLAGS.ROTATION:
          if (Math.abs(this.rotationAngle) == this.movementData[3] * 90) {
            this.rotationAngle = 0;
            rotationData = [];
            this.#rotate(this.movementData, this.movementData.length < 5);
            this.movementData = this.movementData.slice(0, 4);
            this.#updateInterpolationBuffer();
          }
          colors = this.#getPiecesColors(data.sequence);
          this.#updatePieceColorBuffer(colors);
        case Drawer.UPDATE_FLAGS.FORCE_RENDER:
          this.render();
          break;
        case RubixPuzzleDrawer.UPDATE_FLAGS.INTERPOLATION:
          console.log(this.movementData);
          if (rotationData.length == 0) {
            rotationData = [
              this.#getPiecesIsRotating(data.sequence),
              this.#getAvgBounds(data.sequence),
            ];
            this.#updatePieceIsRotatingBuffer(rotationData[0]);
            this.#updateCentersBuffer(rotationData[1]);
          }
          this.rotationAngle +=
            this.rotationSpeed *
            (this.movementData[0] % 2 ? -1 : 1) *
            (this.movementData[2] === "cw" ? 1 : -1);
          this.#updateInterpolationBuffer();
          this.render();
          break;
        default:
          break;
      }
      if (this.rotationAngle == 0) this.updateFlag = Drawer.UPDATE_FLAGS.IDLE;
      else if (Math.abs(this.rotationAngle) == this.movementData[3] * 90)
        this.updateFlag = RubixPuzzleDrawer.UPDATE_FLAGS.ROTATION;
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

    const TEST = new Float32Array(AID_MESH.flat());
    this.createBuffer(
      "aid_mesh_buffer",
      TEST.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.writeBuffer1to1("aid_mesh_buffer", TEST);
  }

  #createBindBuffers() {
    const pieceCount = this.puzzle.pieceCount;
    this.createBuffer(
      "piece_colors",
      Uint32Array.BYTES_PER_ELEMENT * pieceCount * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "piece_positions",
      Float32Array.BYTES_PER_ELEMENT * pieceCount * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "piece_is_rotating",
      Uint32Array.BYTES_PER_ELEMENT * pieceCount,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "rotating_center_avg_ids",
      Uint32Array.BYTES_PER_ELEMENT *
        (this.puzzle.faces[0].corners.length || 1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "rotation_interpolation",
      Float32Array.BYTES_PER_ELEMENT * 2,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "render_state",
      Float32Array.BYTES_PER_ELEMENT * 6,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.createBuffer(
      "colors",
      Float32Array.BYTES_PER_ELEMENT * 4 * this.colors.length,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
  }

  #destroyBuffers() {
    this.getBuffer("piece_colors")?.destroy();
    this.getBuffer("piece_positions")?.destroy();
    this.getBuffer("piece_is_rotating")?.destroy();
    this.getBuffer("rotating_center_avg_ids")?.destroy();
    this.getBuffer("rotation_interpolation")?.destroy();
    this.getBuffer("render_state")?.destroy();
    this.getBuffer("colors")?.destroy();
  }

  #updateCentersBuffer(centers) {
    const CENTERS = new Int32Array(centers.flat());
    this.writeBuffer1to1("rotating_center_avg_ids", CENTERS);
  }

  #updatePiecePositionsBuffer(positions) {
    const PIECE_POSITIONS = new Float32Array(positions.flat());
    this.writeBuffer1to1("piece_positions", PIECE_POSITIONS);
  }

  #updatePieceColorBuffer(colors) {
    const PIECE_COLORS = new Uint32Array(colors.flat());
    this.writeBuffer1to1("piece_colors", PIECE_COLORS);
  }

  #updatePieceIsRotatingBuffer(isRotatings) {
    const PIECE_IS_ROTATING = new Uint32Array(isRotatings.flat());
    this.writeBuffer1to1("piece_is_rotating", PIECE_IS_ROTATING);
  }

  #updateColorsBuffer() {
    const COLOR = new Float32Array(this.colors.flat());
    this.writeBuffer1to1("colors", COLOR);
  }

  #updateRenderStateBuffer() {
    const stateData = [
      this.gpuData.ratio,
      this.#calculateScale(),
      this.#calculatePuzzleCenter(),
      this.cursor.x,
      this.cursor.y,
      this.cursor.z,
    ];

    const RENDER_STATE = new Float32Array(stateData);
    this.writeBuffer1to1("render_state", RENDER_STATE);
  }

  #updateInterpolationBuffer() {
    const ANGLE = new Float32Array([
      this.rotationAngle,
      this.movementData[0] > 2
        ? 5 - this.movementData[0]
        : this.movementData[0],
    ]);
    this.writeBuffer1to1("rotation_interpolation", ANGLE);
  }

  #createMainRenderPipeline() {
    const bindGroup0Layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
          },
        },
      ],
    });

    const bindGroup0 = this.device.createBindGroup({
      label: "zero",
      layout: bindGroup0Layout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer("piece_colors"),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer("piece_positions"),
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.getBuffer("piece_is_rotating"),
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.getBuffer("rotating_center_avg_ids"),
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.getBuffer("rotation_interpolation"),
          },
        },
      ],
    });

    const bindGroup1Layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
      ],
    });

    const bindGroup1 = this.device.createBindGroup({
      layout: bindGroup1Layout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer("render_state"),
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.getBuffer("colors"),
          },
        },
      ],
    });

    this.addBindGroup(bindGroup0);
    this.addBindGroup(bindGroup1);

    return this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroup0Layout, bindGroup1Layout],
      }),
      vertex: {
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
        ],
      },
      fragment: {
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

  #createFaceAidRenderPipeline() {
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {},
        },
        ...Array.from({ length: this.puzzle.faces.length }, (_, i) => {
          return {
            binding: i + 2,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {},
          };
        }),
      ],
    });
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.getBuffer("render_state"),
          },
        },
        {
          binding: 1,
          resource: this.getSampler("sampler_1"),
        },
        ...Array.from({ length: this.puzzle.faces.length }, (_, i) => {
          return {
            binding: i + 2,
            resource: this.getTexture(`face_image_${i}`).createView(),
          };
        }),
      ],
    });
    this.addBindGroup(bindGroup);

    const color = {
      operation: 'add',
      srcFactor: 'dst-alpha',
      dstFactor: 'one-minus-src-alpha',
    }
    const alpha = {
      operation: 'add',
      srcFactor: 'dst-alpha',
      dstFactor: 'one-minus-src-alpha',
    }

    return this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: this.getShader("frag_vert_aid"),
        buffers: [
          {
            arrayStride: 24,
            stepMode: "vertex",
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3",
              },
              {
                shaderLocation: 1,
                offset: 12,
                format: "float32x2",
              },
              {
                shaderLocation: 2,
                offset: 20,
                format: "float32",
              },
            ],
          },
        ],
      },
      fragment: {
        module: this.getShader("frag_vert_aid"),
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat(), blend: {color, alpha} }],
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
      
    let commandEncoder = this.device.createCommandEncoder();
    let passEncoder = commandEncoder.beginRenderPass(
      this.gpuData.renderPassDescriptor
    );

    passEncoder.setPipeline(this.getRenderPipeline("main"));
    passEncoder.setVertexBuffer(0, this.getBuffer("triangles_buffer"));
    passEncoder.setVertexBuffer(1, this.getBuffer("triangle_enum"));
    this.gpuData.bindGroups
      .slice(0, 2)
      .forEach((e, i) => passEncoder.setBindGroup(i, e));
    passEncoder.draw(
      this.getBuffer("triangles_buffer").size /
        (3 * Float32Array.BYTES_PER_ELEMENT),
      this.puzzle.pieceCount
    );

    if (this.showFaceAid) {
      passEncoder.setPipeline(this.getRenderPipeline("aid"));
      passEncoder.setVertexBuffer(0, this.getBuffer("aid_mesh_buffer"));
      passEncoder.setBindGroup(0, this.gpuData.bindGroups[2]);
      passEncoder.draw(
        this.getBuffer("aid_mesh_buffer").size / (6 * Float32Array.BYTES_PER_ELEMENT)
      );
    }
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  #getAvgBounds(sequence) {
    const [face, depth] = this.movementData;
    const boundIndices = [];
    const set = new Set(
      this.puzzle.faces[face].corners.map(
        (e) => e.getDeepPiece(5 - face, depth).id
      )
    );

    for (let i = 0; i < sequence.length; i++) {
      const id = sequence[i].id;
      if (set.has(id)) {
        boundIndices.push(i);
        set.delete(id);
      }
      if (!set.size) return boundIndices;
    }
    boundIndices.push(-1);
    return boundIndices;
  }

  #getPiecesColors(sequence) {
    const allColors = [];
    for (const piece of sequence) {
      allColors.push(piece.getColoring());
    }
    return allColors;
  }

  #getPiecesIsRotating(sequence) {
    const [face, depth] = this.movementData;
    const set = new Set(
      this.puzzle.faces[face]
        .march(
          this.puzzle.faces[face].corners[0].getDeepPiece(5 - face, depth),
          1
        )
        .map((e) => e.id)
    );

    return sequence.map((e) => (set.has(e.id) ? 1 : 0));
  }

  #getPiecesVisitSequenceAndPositions() {
    const length = this.puzzle.length;

    const pieceMap = new Map();
    const removalQueue = [
      length === 1
        ? this.puzzle.faces[5].centers[0]
        : this.puzzle.faces[5].corners[3],
    ];
    const sequence = [];
    const positions = [];

    while (removalQueue.length) {
      const currPiece = removalQueue.shift();
      sequence.push(currPiece);

      if (!pieceMap.has(currPiece.id)) {
        const index = positions.push([0, 0, 0, 1]) - 1;
        pieceMap.set(currPiece.id, index);
      }

      for (const [faceIndex, face] of currPiece.faceData.entries()) {
        if (face?.id === undefined || pieceMap.has(face.id)) continue;
        const delta = this.#getPieceFaceDelta(faceIndex);
        const position = positions[pieceMap.get(currPiece.id)].map(
          (e, i) => e + (i < 3 ? delta[i] : 0)
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
    return 0.8 / this.puzzle.length;
  }

  #calculatePuzzleCenter() {
    return (this.puzzle.length - 1) / 2;
  }

  #incrementPuzzleSize(delta) {
    const newLength = this.puzzle.length + delta;
    if (newLength < 1 || newLength > 50) return;
    this.puzzle = new RubixPuzzle(newLength);
    this.updateFlag = Drawer.UPDATE_FLAGS.RESET;
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

  #rotate([face, depth, direction, count], store = true) {
    if (face === undefined) return;
    const ids = this.puzzle.rotate(face, depth, direction, count, store);
  }

  #startInterpolation() {
    this.updateFlag = RubixPuzzleDrawer.UPDATE_FLAGS.INTERPOLATION;
  }

  #unscramble() {
    const interval = setInterval(() => {
      if (this.updateFlag != Drawer.UPDATE_FLAGS.IDLE) return;
      const move = this.puzzle.reverseLastMove();
      if (!move) {
        clearInterval(interval);
        return;
      }
      this.movementData = move;
      this.#startInterpolation();
    }, 300);
  }
}