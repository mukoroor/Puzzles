export const move_eval_shader = (cubeDim, moves) => {
    console.log(moves)
    const movesCount = moves.length;
    const movesString = movesToString(moves);
    // console.log(movesString)
    return /*wgsl*/ `
    const adj = array(
        array(1, 2, 4, 3),
        array(0, 3, 5, 2),
        array(0, 1, 5, 4),
    );

    const dimension = ${cubeDim};
    const moves: array<CubeMove, ${movesCount}>  = array(${movesString});

    
    alias Face = array<array<u32, dimension>, dimension>;
    alias Cube = array<Face, 6>;

    struct CubeMove {
        face: u32,
        depth: u32,
        rotCount: u32,
    }

    @group(0) @binding(0)
    var<storage, read_write> cubes: array<Cube>;
    @group(0) @binding(1)
    var<storage, read_write> output: array<array<f32, 6>, ${movesCount}>;
    @group(0) @binding(2)
    var<storage, read_write> cubesOut: array<Cube>;


    @compute @workgroup_size(1)
    // @compute @workgroup_size(${cubeDim}, 3, 3)
    fn main(@builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u) {
        let index: u32 = global_id.x;
        if (index >= ${movesCount}) {
            return;
        }

        rotateCube(index, &(cubes[0]), &(cubesOut[index]));
        for (var i: u32  = 0; i < 6; i++) {
            output[index][i] = scoreDP(&(cubesOut[index][i]));
        }
    }


    fn scoreDP(arrP: ptr<storage, Face, read_write>) -> f32 {
        var len = 6;
        var idCache = array<i32, dimension>();
        var idCurr = array<i32, dimension>();
        var scoring = array<i32, 6>();
        var count = 0;
        var ids = 1;

        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                var val = (*arrP)[i][j];
                if ((i != 0 && (*arrP)[i - 1][j] == val) && (j != 0 && (*arrP)[i][j - 1] == val) && idCache[j] != idCurr[j - 1]) {
                    idCurr[j] = idCurr[j - 1];
                    scoring[val]--;
                    count--;
                } else if (i != 0 && (*arrP)[i - 1][j] == val) {
                    idCurr[j] = idCache[j];
                } else if (j != 0 && (*arrP)[i][j - 1] == val) {
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

    fn nextCube(cubePtr: ptr<storage, Cube, read_write>, movePtr: ptr<function, CubeMove>) -> Cube {
        return Cube();
    }

    fn rotate90(arrPIn: ptr<storage, Face, read_write>, arrPOut: ptr<storage, Face, read_write>) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                (*arrPOut)[dimension - j - 1][i] = (*arrPIn)[i][j];
            }
        }
    }

    fn rotate180(arrPIn: ptr<storage, Face, read_write>, arrPOut: ptr<storage, Face, read_write>) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                (*arrPOut)[dimension - i - 1][dimension - j - 1] = (*arrPIn)[i][j];
            }
        }
    }

    fn rotateN90(arrPIn: ptr<storage, Face, read_write>, arrPOut: ptr<storage, Face, read_write>) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                (*arrPOut)[j][dimension - i - 1] = (*arrPIn)[i][j];
            }
        }
    }

    fn rotateFace(arrPIn: ptr<storage, Face, read_write>, arrPOut: ptr<storage, Face, read_write>, rotCount: u32) {
        switch rotCount {
            case 1: {
                rotate90(arrPIn, arrPOut);                                                 
            }
            case 2: {
                rotate180(arrPIn, arrPOut);                                                 
            }
            case 3: {
                rotateN90(arrPIn, arrPOut);                                                 
            }
            default: {}
        }
    }

    fn copy1to1(arrPIn: ptr<storage, Face, read_write>, arrPOut: ptr<storage, Face, read_write>) {
        for (var i: u32 = 0; i < dimension; i++) {
            for (var j: u32 = 0; j < dimension; j++) {
                (*arrPOut)[i][j] = (*arrPIn)[i][j];
            }
        }
    }


    fn rotateResidualFaces(moveInd: u32, cubePtrIn: ptr<storage, Cube, read_write>, cubePtrOut: ptr<storage, Cube, read_write>) {
        var cubeMove = moves[moveInd];
        for (var f: u32 = 0; f < 4; f++) {
            let curr = adj[cubeMove.face][f];
            let next = adj[cubeMove.face][(f + cubeMove.rotCount) % 4];
            for (var i: u32 = 0; i < dimension; i++) {
                if (cubeMove.face == 0) {
                    (*cubePtrOut)[next][cubeMove.depth][i] = (*cubePtrIn)[curr][cubeMove.depth][i];
                } else if (cubeMove.face == 1) {
                    if (curr % 2 == 1) {
                        if (next % 2 == 1) {
                            (*cubePtrOut)[next][i][dimension - 1 - cubeMove.depth] = (*cubePtrIn)[curr][i][dimension - 1 - cubeMove.depth];
                        } else {
                            (*cubePtrOut)[next][i][cubeMove.depth] = (*cubePtrIn)[curr][i][dimension - 1 - cubeMove.depth];
                        }
                    } else {
                        if (next % 2 == 1) {
                            (*cubePtrOut)[next][i][dimension - 1 - cubeMove.depth] = (*cubePtrIn)[curr][i][cubeMove.depth];
                        } else {
                            (*cubePtrOut)[next][i][cubeMove.depth] = (*cubePtrIn)[curr][i][cubeMove.depth];
                        }
                    }
                } else if (cubeMove.face == 2) {
                    if (curr == 0 || curr == 5) {
                        if (next == 0 || next == 5) {
                            (*cubePtrOut)[next][dimension - 1 - cubeMove.depth][i] = (*cubePtrIn)[curr][dimension - 1 - cubeMove.depth][i];
                        } else if (next == 1) {
                            (*cubePtrOut)[next][dimension - 1 - i][dimension - 1 - cubeMove.depth] = (*cubePtrIn)[curr][dimension - 1 - cubeMove.depth][i];
                        } else if (next == 4) {
                            (*cubePtrOut)[next][i][cubeMove.depth] = (*cubePtrIn)[curr][dimension - 1 - cubeMove.depth][i];
                        }
                    } else if (curr == 1) {
                        if (next == 0 || next == 5) {
                            (*cubePtrOut)[next][dimension - 1 - cubeMove.depth][i] = (*cubePtrIn)[curr][dimension - 1 - i][dimension - 1 - cubeMove.depth];
                        } else if (next == 4) {
                            (*cubePtrOut)[next][dimension - 1 - cubeMove.depth][i] = (*cubePtrIn)[curr][dimension - 1 - i][dimension - 1 - cubeMove.depth];
                        }
                    } else if (curr == 4) {
                        if (next == 0 || next == 5) {
                            (*cubePtrOut)[next][dimension - 1 - cubeMove.depth][i] = (*cubePtrIn)[curr][i][cubeMove.depth];
                        } else if (next == 1) {
                            (*cubePtrOut)[next][dimension - 1 - i][dimension - 1 - cubeMove.depth] = (*cubePtrIn)[curr][i][cubeMove.depth];
                        }
                    }
                }
            }
        }
    }

    fn rotateCube(moveInd: u32, cubePtrIn: ptr<storage, Cube, read_write>, cubePtrOut: ptr<storage, Cube, read_write>) {
        var faceRot = moves[moveInd].face;
        var count = moves[moveInd].rotCount;
        if (moves[moveInd].depth == dimension - 1) {
            faceRot = 5 - faceRot;
            count = 4 - count;
        }

        for (var i: u32 = 0; i < 6; i++) {
            if (i == faceRot) {rotateFace(&(*cubePtrIn)[i], &(*cubePtrOut)[i], count);}
            else {copy1to1(&(*cubePtrIn)[i], &(*cubePtrOut)[i]);}
        }

        rotateResidualFaces(moveInd, cubePtrIn, cubePtrOut);
    }
`;
}

function movesToString(moves) {
    return moves.map(e => {
        return `CubeMove(${e.join(',')})`
    }).join(', ')
}
