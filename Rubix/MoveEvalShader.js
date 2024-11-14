export const move_eval_shader = (cubeDim, moves) => {
    // console.log(moves)
    const movesCount = moves.length;
    const movesString = movesToString(moves);
    // console.log(movesString)
    return /*wgsl*/ `
    const adj = array<array<u32, 4>, 3>(
        array(1, 2, 4, 3),
        array(0, 3, 5, 2),
        array(0, 1, 5, 4),
    );

    const dimension = ${cubeDim};
    const dim_sq = dimension * dimension;
    const moves: array<CubeMove, ${movesCount}>  = array(${movesString});

    struct CubeMove {
        face: u32,
        depth: u32,
        rotCount: u32,
    }

    @group(0) @binding(0)
    var<storage, read_write> cubesU32: array<atomic<u32>>;
    @group(0) @binding(1)
    var<storage, read_write> output: array<array<f32, 6>>;
    @group(0) @binding(2)
    var<storage, read_write> cubesOut: array<atomic<u32>>;


    @compute @workgroup_size(1)
    fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u, @builtin(num_workgroups) num_workgroups: vec3<u32>) {

        let index =  
            workgroup_id.x +
            workgroup_id.y * num_workgroups.x +
            workgroup_id.z * num_workgroups.x * num_workgroups.y;
        if (index >= arrayLength(&output)) {
            return;
        }

        // rotateCube(23, index / ${movesCount}, index);
        rotateCube(index % ${movesCount}, index / ${movesCount}, index);
        for (var i: u32  = 0; i < 6; i++) {
            // output[index][0] = f32(index % ${movesCount});
            output[index][i] = scoreDP(index, i);
            // output[index][1] = f32(index);
        }
    }


    fn scoreDP(cubeInd: u32, faceInd: u32) -> f32 {
        var len = 6;
        var idCache = array<i32, dimension>();
        var idCurr = array<i32, dimension>();
        var scoring = array<i32, 6>();
        var count = 0;
        var ids = 1;

        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                var val = getPiece(&cubesOut, cubeInd, faceInd, i, j);
                var diffRowVal = getPiece(&cubesOut, cubeInd, faceInd, i - 1, j);
                var diffColVal = getPiece(&cubesOut, cubeInd, faceInd, i, j - 1);
                if ((i != 0 && diffRowVal == val) && (j != 0 && diffColVal == val) && idCache[j] != idCurr[j - 1]) {
                    idCurr[j] = idCurr[j - 1];
                    scoring[val]--;
                    count--;
                } else if (i != 0 && diffRowVal == val) {
                    idCurr[j] = idCache[j];
                } else if (j != 0 && diffColVal == val) {
                    idCurr[j] = idCurr[j - 1];
                } else {
                    idCurr[j] = ids;
                    ids++;
                    scoring[val]++;
                    count++;
                }
            }
            var temp = idCache;
            idCache = idCurr;
            idCurr = temp;
        }

        var tot: i32 = 0;
        var score: i32 = 0;
        for (var i: u32 = 0; i < 6; i++) {
            tot += scoring[i];
            score += scoring[i] * scoring[i];
        }
        return f32(score) / f32(tot * tot);
    }

    fn rotate90(cubeIndIn: u32, cubeIndOut: u32, faceInd: u32) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                // setPiece(cubeIndOut, faceInd, dimension - j - 1, i, 1);
                setPiece(cubeIndOut, faceInd, dimension - j - 1, i, getPiece(&cubesU32, cubeIndIn, faceInd, i, j));
            }
        }
    }

    fn rotate180(cubeIndIn: u32, cubeIndOut: u32, faceInd: u32) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                // setPiece(cubeIndOut, faceInd, dimension - i - 1, dimension - j - 1, 1);
                setPiece(cubeIndOut, faceInd, dimension - i - 1, dimension - j - 1, getPiece(&cubesU32, cubeIndIn, faceInd, i, j));
            }
        }
    }

    fn rotateN90(cubeIndIn: u32, cubeIndOut: u32, faceInd: u32) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                // setPiece(cubeIndOut, faceInd, j, dimension - i - 1, 1);
                setPiece(cubeIndOut, faceInd, j, dimension - i - 1, getPiece(&cubesU32, cubeIndIn, faceInd, i, j));
            }
        }
    }

    fn rotateFace(cubeIndIn: u32, cubeIndOut: u32, faceInd: u32, rotCount: u32) {
        switch rotCount {
            case 1: {
                rotate90(cubeIndIn, cubeIndOut, faceInd);                                                 
            }
            case 2: {
                rotate180(cubeIndIn, cubeIndOut, faceInd);                                                 
            }
            case 3: {
                rotateN90(cubeIndIn, cubeIndOut, faceInd);                                                 
            }
            default: {
                copyFace(cubeIndIn, cubeIndOut, faceInd);
            }
        }
    }

    fn copyFace(cubeIndIn: u32, cubeIndOut: u32, faceInd: u32) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                // if (faceInd == 5) {
                    // setPiece(cubeIndOut, faceInd, i, j, 7);
                    // continue;
                // }
                setPiece(cubeIndOut, faceInd, i, j, getPiece(&cubesU32, cubeIndIn, faceInd, i, j));
            }
        }
    }


    fn rotateResidualFaces(moveInd: u32, cubeIndIn: u32, cubeIndOut: u32) {
        var cubeMove = moves[moveInd];
        for (var f: u32 = 0; f < 4; f++) {
            let curr = adj[cubeMove.face][f];
            let next = adj[cubeMove.face][(f + cubeMove.rotCount) % 4];
            for (var i: u32 = 0; i < dimension; i++) {
                if (cubeMove.face == 0) {
                    setPiece(cubeIndOut, next, cubeMove.depth, i, getPiece(&cubesU32, cubeIndIn, curr, cubeMove.depth, i));
                } else if (cubeMove.face == 1) {
                    if (curr % 2 == 1) {
                        if (next % 2 == 1) {
                            setPiece(cubeIndOut, next, i, dimension - 1 - cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, i, dimension - 1 - cubeMove.depth));
                        } else {
                            setPiece(cubeIndOut, next, i, cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - i, dimension - 1 - cubeMove.depth));
                        }
                    } else {
                        if (next % 2 == 1) {
                            setPiece(cubeIndOut, next, dimension - 1 - i, dimension - 1 - cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, i, cubeMove.depth));
                        } else {
                            setPiece(cubeIndOut, next, i, cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, i, cubeMove.depth));
                        }
                    }
                } else if (cubeMove.face == 2) {
                    if (curr == 0 || curr == 5) {
                        if (next == 0 || next == 5) {
                            setPiece(cubeIndOut, next, dimension - 1 - cubeMove.depth, i, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - cubeMove.depth, i));
                        } else if (next == 1) {
                            setPiece(cubeIndOut, next, dimension - 1 - i, dimension - 1 - cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - cubeMove.depth, i));
                        } else if (next == 4) {
                            setPiece(cubeIndOut, next, i, cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - cubeMove.depth, i));
                        }
                    } else if (curr == 1) {
                        if (next == 0 || next == 5) {
                            setPiece(cubeIndOut, next, dimension - 1 - cubeMove.depth, i, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - i, dimension - 1 - cubeMove.depth));
                        } else if (next == 4) {
                            setPiece(cubeIndOut, next, i, cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, dimension - 1 - i, dimension - 1 - cubeMove.depth));
                        }
                    } else if (curr == 4) {
                        if (next == 0 || next == 5) {
                            setPiece(cubeIndOut, next, dimension - 1 - cubeMove.depth, i, getPiece(&cubesU32, cubeIndIn, curr, i, cubeMove.depth));
                        } else if (next == 1) {
                            setPiece(cubeIndOut, next, dimension - 1 - i, dimension - 1 - cubeMove.depth, getPiece(&cubesU32, cubeIndIn, curr, i, cubeMove.depth));
                        }
                    }
                }
            }
        }
    }

    fn rotateCube(moveInd: u32, cubeIndIn: u32, cubeIndOut: u32) {
        var faceRot = moves[moveInd].face;
        var count = moves[moveInd].rotCount;
        if (moves[moveInd].depth == dimension - 1) {
            faceRot = 5 - faceRot;
            count = 4 - count;
        }

        for (var i: u32 = 0; i < 6; i++) {
            if (i == faceRot && moves[moveInd].depth % (dimension - 1) == 0) {rotateFace(cubeIndIn, cubeIndOut, i, count);}
            else {copyFace(cubeIndIn, cubeIndOut, i);}
        }

        rotateResidualFaces(moveInd, cubeIndIn, cubeIndOut);
    }

    fn getPiece(cubesArr: ptr<storage, array<atomic<u32>>, read_write>, cubeIndex: u32, faceIndex: u32, pieceRow: u32, pieceCol: u32) -> u32 {
        var index = 3 * ((6 * cubeIndex + faceIndex) * dim_sq + pieceRow * dimension + pieceCol);
        var flr = (index - index % 32) / 32;
        var rem = (index + 3) % 32;

        var relInd = index - flr * 32;
        var val = ((7u << relInd) & atomicLoad(&(*cubesArr)[flr])) >> relInd;
        if (rem == 1) {
            val = val | ((atomicLoad(&(*cubesArr)[flr + 1]) & (1u)) << 2u);
        } else if (rem == 2) {
            val = val | ((atomicLoad(&(*cubesArr)[flr + 1]) & (3u)) << 1u);
        }

        return val;
    }

    fn setPiece(cubeIndex: u32, faceIndex: u32, pieceRow: u32, pieceCol: u32, val: u32) {
        var index = 3 * ((6 * cubeIndex + faceIndex) * dim_sq + pieceRow * dimension + pieceCol);
        var flr = (index - index % 32) / 32;
        var rem = (index + 3) % 32;

        var relInd = index - flr * 32;
        atomicAnd(&cubesOut[flr], ~(7u << relInd));
        atomicOr(&cubesOut[flr] , (val << relInd));
        // atomicOr(&cubesOut[flr] , (~(7u << relInd) & atomicLoad(&cubesOut[flr])) | (val << relInd));
        if (rem == 1) {
            atomicAnd(&cubesOut[flr + 1], ~(1u));
            atomicOr(&cubesOut[flr + 1] , (val >> 2u));
            // atomicOr(&cubesOut[flr + 1] , (atomicLoad(&cubesOut[flr + 1]) & ~(1u)) | (val >> 2u));
        } else if (rem == 2) {
            atomicAnd(&cubesOut[flr + 1], ~(3u));
            atomicOr(&cubesOut[flr + 1] , (val >> 1u));
            // atomicOr(&cubesOut[flr + 1] , (atomicLoad(&cubesOut[flr + 1]) & ~(3u)) | (val >> 1u));
        }
    }
`;
}

function movesToString(moves) {
    return moves.map(e => {
        return `CubeMove(${e.join(',')})`
    }).join(', ')
}
