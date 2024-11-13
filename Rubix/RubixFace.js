import RubixPiece from "./RubixPiece.js";

export default class RubixFace {
  static FACE_ADJACENCY = Object.freeze({
    0: [1, 2, 4, 3],
    1: [0, 2, 5, 3],
    2: [0, 4, 5, 1],
    3: [0, 4, 5, 1],
    4: [0, 3, 5, 2],
    5: [4, 2, 1, 3],
  });
  id;
  #lengthFunc;
  corners = [];
  edges = [];
  interiors = [];
  centers = [];
  #memoize = {
  }

  constructor(id, lengthFunc) {
    this.id = id;
    this.#lengthFunc = lengthFunc;
  }

  get length() {
    return this.#lengthFunc();
  }

  addCenter(faceCenter) {
    this.centers.push(faceCenter);
  }

  addCorner(faceCorner) {
    this.corners.push(faceCorner);
  }

  addEdge(faceEdge) {
    this.edges.push(faceEdge);
  }

  addInterior(faceInterior) {
    this.interiors.push(faceInterior);
  }

  rotate(depth = 0, direction = this.#cornerOrientation, count = 0) {
    count = count % 4;

    if (!count || depth > this.length - 1) return;

    const depthJoint = 5 - this.id;
    const nextBoundIndex =
      (direction === this.#cornerOrientation ? count : 4 - count) % 4;

    let currBound = this.corners[0].getDeepPiece(depthJoint, depth),
      nextBound = this.corners[nextBoundIndex].getDeepPiece(depthJoint, depth);

    const start = this.march(currBound, 1);
    const finish = this.march(nextBound, (nextBoundIndex + 1) % 4);

    const constFace = depth + 1 === this.length ? depthJoint : this.id;

    const updatedPieceData = start.map((e, i) =>
      finish[i].generateNextData(e, constFace)
    );

    finish.forEach((e, i) => {
      const nextData = updatedPieceData[i];
      e.faceData = nextData;
    });
  }

  fill() {
    let max = this.length - 2;

    let rings = [this.#buildCornerRing()];
    
    if (this.length < 3) {
      rings[0].forEach(e => this.addCenter(e));
      return;
    };

    for (let i = max; i > 0; i -= 2) {
      let ring = this.#buildRing(i);
      if (i < 3) {
        ring.forEach(e => this.addCenter(e));
      };
      rings.unshift(ring);
    }

    for (let i = 1; i < rings.length; i++) {
      this.#merge(rings[i - 1], rings[i]);
    }
  }

  #buildRing(length) {
    if (length === 1) {
      this.center = new RubixPiece(`${this.id}`, `${this.id}`, "CENTER");
      return [this.center];
    }

    const maxChain = length - 1;
    const faceJoints = this.#jointPath;
    const cycle = Array(4 * maxChain);

    for (let i = 0; i < 4; i++) {
      let jointStart = faceJoints[i],
        jointEnd = 5 - jointStart;
      for (let j = 0; j < maxChain; j++) {
        let piece = new RubixPiece(`${this.id}`, `${this.id}`, "INTERIOR"),
          prevPiece =
            cycle[(cycle.length + maxChain * i + j - 1) % cycle.length];
        if (i || j) {
          this.addInterior(piece);
          RubixPiece.join(piece, prevPiece, jointStart, jointEnd);
        }

        cycle[maxChain * i + j] = piece;
      }
    }
    RubixPiece.join(
      cycle[0],
      cycle[4 * maxChain - 1],
      faceJoints[0],
      5 - faceJoints[0]
    );

    return cycle;
  }

  march(piece, startingFaceIndex = RubixFace.FACE_ADJACENCY[this.id][0]) {
    const path = [piece];
    const s = new Set([piece.id]);

    const jointPath = this.#jointPath;
    const len = this.length * this.length;

    while (path.length !== len) {
      const nextPiece = path.at(-1).faceData[jointPath[startingFaceIndex]];
      if (nextPiece === undefined) break;
      else if (s.has(nextPiece.id) || typeof nextPiece === "number") {
        startingFaceIndex = (startingFaceIndex + 1) % 4;
        continue;
      }

      path.push(nextPiece);
      s.add(nextPiece.id);

      if (nextPiece.type === "CORNER") {
        startingFaceIndex = (startingFaceIndex + 1) % 4;
      }
    }
    return path;
  }

  #buildCornerRing() {
    const joints = this.#jointPath;
    const targetCornerActivations = [this.id, joints[0], joints[1]];
    let target = this.corners.find((corner) =>
      targetCornerActivations.every(
        (activation) => typeof corner.getFace(activation) === "number"
      )
    );

    const outerChain = this.length - 1;
    const cycle = Array(4 * outerChain);

    for (let i = 0; i < 4; i++) {
      let joint = 5 - joints[i];
      for (let j = 0; j < outerChain; j++) {
        target = target.getFace(joint);
        cycle[outerChain * i + j] = target;
      }
    }

    return cycle;
  }

  #merge(innerRing, outerRing) {
    const mergeJoints = this.#jointPath;
    const innerSize = Math.floor(innerRing.length / 4) + 1;
    mergeJoints.push(mergeJoints.shift(), null);

    innerRing.unshift(innerRing.pop());

    let innerPointer = 0,
      outerPointer = 0,
      counter = 0;
    let face = mergeJoints.shift(),
      flag = false;

    while (innerPointer <= innerRing.length && face !== null) {
      counter++;
      RubixPiece.join(
        innerRing[innerPointer++ % innerRing.length],
        outerRing[outerPointer++],
        face,
        5 - face
      );
      if (!(counter % innerSize) && !flag) {
        innerPointer--;
        outerPointer++;
        face = mergeJoints.shift();
        flag = innerSize !== 1;
      } else {
        flag = false;
      }
    }
  }

  get #jointPath() {
    return [...RubixFace.FACE_ADJACENCY[this.id]];
  }

  get #cornerOrientation() {
    return 8 % this.id == 0 ? 'cw': 'ccw';
  }

  toString() {
    return `~~${this.id}~~
                CORNERS: ${this.corners.map((corner) => corner.toString())}\n
                EDGES: ${this.edges.map((edge) => edge.toString())}\n
                INTERIORS: ${this.interiors.map((interior) =>
                  interior.toString()
                )}\n\n`;
  }

  score() {
    console.time('reg')
    const m = {}
    const f = p => {
      let key = p.faceData[this.id];
      if (m[key]) m[key]++;
      else m[key] = 1;
    }
    this.corners.forEach(f);
    this.edges.forEach(f);
    this.interiors.forEach(f);
    this.centers.forEach(f);

    let sum = 0;
    let pieceCount = this.length * this.length;
    for (const val of Object.values(m)) {
      sum += Math.pow(val / pieceCount , 2)
    }
    console.timeEnd('reg')
    console.log(Object.values(m), 'reg score');
    return sum;
  }

  scoreDFS() {
    console.time('dfs')
    const visited = new Set();

    const types = {}

    let q = [this.corners[0]]
    while (q.length != 0) {
      let curr = q.shift()
      let key = curr.faceData[this.id];
      if (visited.has(curr.id)) {
        types[key]++;
        continue;
      }
      visited.add(curr.id);
      if (types[key] != undefined) types[key]++;
      else types[key] = 1;

      for (const v of RubixFace.FACE_ADJACENCY[this.id]) {
        if (typeof curr.faceData[v] != 'number' && !visited.has(curr.faceData[v].id)) {
          if (curr.faceData[v].faceData[this.id] == key) {
            types[key]--;
          }
          q.unshift(curr.faceData[v]);
        }
      }
    }
    
    let tot = Object.values(types).reduce((a, c) => a + c), sum = 0;
    for (const val of Object.values(types)) {
      sum += Math.pow(val / tot , 2)
    }
    console.timeEnd('dfs')
    console.log(Object.values(types), 'dfs score');
    return sum;
  }

  to2DArray() {
    if (this.#memoize.twoD) return this.#memoize.twoD;
    if (this.length == 1) return [[this.centers[0]]];

    const arr = Array.from({length: this.length}, () => Array(this.length));

    let currRow = this.corners[this.id == 3 ? 1: 0];

    let rowJoint = currRow.activations[this.id].next.face, colJoint = currRow.activations[rowJoint].next.face;
    rowJoint = 5 - rowJoint;
    colJoint = 5 - colJoint;
    for (let i = 0; i < arr.length; i++) {
      let currEl  = currRow;
      for (let j = 0; j < arr.length; j++) {
        arr[i][j] = currEl;
        currEl = currEl.faceData[colJoint];
      }
      currRow = currRow.faceData[rowJoint];
    }
    this.#memoize.twoD = arr;
    return arr;
  }

  toBinary() {
    const twoD = this.to2DArray().map(e => e.map(p => p.faceData[this.id]));
    return twoD.flat(2).map(e => e.toString(2).padStart(3, '0')).reverse().join('');
  }

  scoreDP(arr) {
    // console.time('dp')
    const scoring = Array(6).fill(0);
    const twoD = arr || this.to2DArray().map(e => e.map(c => c.faceData[this.id]));

    let count = 0, ids = 1;
    let idCache = Array(twoD.length).fill(0);
    let idCurr = Array(twoD.length).fill(0);

    for (let i = 0; i < twoD.length; i++) {
      for (let j = 0; j < twoD.length; j++) {
        const val = twoD[i][j];
        if ((i != 0 && twoD[i - 1][j] == val) && (j != 0 && twoD[i][j - 1] == val) && idCache[j] != idCurr[j - 1]) {
          idCurr[j] = idCurr[j - 1];
          scoring[val]--;
          count--;
        } else if (i != 0 && twoD[i - 1][j] == val) {
          idCurr[j] = idCache[j];
        } else if (j != 0 && twoD[i][j - 1] == val) {
          idCurr[j] = idCurr[j - 1];
        } else {
          idCurr[j] = ids++;
          scoring[val]++;
          count++;
        }
      }
      idCache = idCurr;
    }
    let tot = scoring.reduce((a, c) => a + c), sum = 0;
    for (const val of scoring) {
      sum += Math.pow(val / tot , 2)
    }
    // console.timeEnd('dp')
    // console.log(scoring, tot, 'dp score')
    return sum;
  }
}
