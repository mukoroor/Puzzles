import { requestWGPUDevice } from "../NeuralNetwork/Tensor/DeviceDecorators/index.js";
import matrixMultiplication from "../NeuralNetwork/Tensor/OperationsGPU.js";
import * as Tensor from '../NeuralNetwork/Tensor/Tensor.js'

document.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("runMatrixMultiplyBtn");
  const sizeInput = document.getElementById("matrixSize");
  const warmupInput = document.getElementById("warmupRuns");
  const resultsDiv = document.getElementById("flopsResults");

  runBtn.addEventListener("click", async () => {
    const size = parseInt(sizeInput.value, 10);
    const warmupRuns = parseInt(warmupInput.value, 10);

    const stats = await runMatrixMultiplyWithStats(size, warmupRuns);
    resultsDiv.innerHTML = `
      <b>Matrix Size:</b> ${size}x${size}<br>
      <b>Warmup Runs:</b> ${warmupRuns}<br>
      <b>Fastest:</b> ${stats.fastest.toFixed(2)} ms<br>
      <b>Slowest:</b> ${stats.slowest.toFixed(2)} ms<br>
      <b>Average:</b> ${stats.average.toFixed(2)} ms<br>
      <b>Median:</b> ${stats.median.toFixed(2)} ms<br>
      <b>FLOPS:</b> ${stats.flops.toExponential(2)}
    `;
  });
});

// UI for accuracy test
document.addEventListener("DOMContentLoaded", () => {
  const testBtn = document.createElement("button");
  testBtn.textContent = "Test Matrix Multiplication Accuracy";
  document.body.appendChild(testBtn);

  const accuracyDiv = document.createElement("div");
  accuracyDiv.id = "accuracyResults";
  document.body.appendChild(accuracyDiv);

  testBtn.addEventListener("click", async () => {
    // Simple 2x2 matrices for accuracy test
    const A = Tensor.from([1, 2, 3, 4], [2, 2], Float16Array);
    const B = Tensor.from([5, 6, 7, 8], [2, 2], Float16Array);

    // const A = Tensor.from([1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4], Float16Array);
    // const B = Tensor.from([5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4], Float16Array);


    // Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    const expected = [
      [1*5+2*7, 1*6+2*8],
      [3*5+4*7, 3*6+4*8]
    ].flat();

    const wgpu_device = await requestWGPUDevice();
    const A_gpu = await Tensor.sendToDevice(A, wgpu_device);
    const B_gpu = await Tensor.sendToDevice(B, wgpu_device);

    const resultTensor = await matrixMultiplication(A_gpu, B_gpu);
    const result = (await Tensor.sendToDevice(resultTensor)).data;

    const isAccurate = equals(result, expected);

    accuracyDiv.innerHTML = `
      <b>Matrix A:</b> [${A.data}]<br>
      <b>Matrix B:</b> [${B.data}]<br>
      <b>Expected:</b> [${expected}]<br>
      <b>Result:</b> [${result}]<br>
      <b>Accurate:</b> ${isAccurate ? "✅" : "❌"}
    `;
  });
});

// UI for customizable accuracy test
document.addEventListener("DOMContentLoaded", () => {
  const accuracyControls = document.createElement("div");
  accuracyControls.innerHTML = `
    <label>Rows A: <input type="number" id="rowsA" value="5" min="1" style="width:50px"></label>
    <label>Cols A / Rows B: <input type="number" id="colsA" value="5" min="1" style="width:50px"></label>
    <label>Cols B: <input type="number" id="colsB" value="5" min="1" style="width:50px"></label>
    <button id="customTestBtn">Test Matrix Multiplication Accuracy</button>
    <div id="customAccuracyResults"></div>
  `;
  document.body.appendChild(accuracyControls);

  document.getElementById("customTestBtn").addEventListener("click", async () => {
    const rowsA = parseInt(document.getElementById("rowsA").value, 10);
    const colsA = parseInt(document.getElementById("colsA").value, 10);
    const colsB = parseInt(document.getElementById("colsB").value, 10);

    // Fill A and B with simple sequential numbers for easy checking
    const A_data = Array.from({length: rowsA * colsA}, (_, i) => i + 1);
    const B_data = Array.from({length: colsA * colsB}, (_, i) => i + 1);

    const A = Tensor.from(A_data, [rowsA, colsA], Float16Array);
    console.log('b')
    const B = Tensor.from(B_data, [colsA, colsB], Float16Array);

    // Compute expected result on CPU
    const expected = [];
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += A_data[i * colsA + k] * B_data[k * colsB + j];
        }
        expected.push(sum);
      }
    }

    const wgpu_device = await requestWGPUDevice();
    const A_gpu = await Tensor.sendToDevice(A, wgpu_device);
    const B_gpu = await Tensor.sendToDevice(B, wgpu_device);

    const resultTensor = await matrixMultiplication(A_gpu, B_gpu);
    const result = await Tensor.sendToDevice(resultTensor);

    const isAccurate = equals(result.data, expected);

    document.getElementById("customAccuracyResults").innerHTML = `
      <b>Matrix A:</b> [${A_data}]<br>
      <b>Matrix B:</b> [${B_data}]<br>
      <b>Expected:</b> [${expected}]<br>
      <b>Result:</b> [${result.data}]<br>
      <b>Accurate:</b> ${isAccurate ? "✅" : "❌"}
    `;
  });
});

async function runMatrixMultiplyWithStats(size, warmupRuns) {
  const wgpu_device = await requestWGPUDevice();

  // Generate random matrices
  const A = Tensor.from([...seq(0, size * size)], [size, size], Float16Array);
  const B = Tensor.from([...seq(0, size * size)], [size, size], Float16Array);

  const A_gpu = await Tensor.sendToDevice(A, wgpu_device);
  const B_gpu = await Tensor.sendToDevice(B, wgpu_device);

  let times = [];
  for (let i = 0; i < warmupRuns; i++) {
    const start = performance.now();
    await matrixMultiplication(A_gpu, B_gpu);
    const end = performance.now();
    times.push(end - start);
  }

  times.sort((a, b) => a - b);
  const fastest = times[0];
  const slowest = times[times.length - 1];
  const average = times.reduce((a, b) => a + b, 0) / times.length;
  const median = times.length % 2 === 0
    ? (times[times.length / 2 - 1] + times[times.length / 2]) / 2
    : times[Math.floor(times.length / 2)];

  // FLOPS calculation: 2 * n^3 / (average_time_in_seconds)
  const flops = (2 * size * size * size) / (average / 1000);

  return { fastest, slowest, average, median, flops };
}

function accuracy(y, yPred) {
  let a = 0;
  for (let i = 0; i < Math.min(y.length, yPred.length); i++) {
    if (equals(y[i], yPred[i])) a++;
  }
  return a / Math.min(y.length, yPred.length);
}

function equals(arr1, arr2) {
  if (arr1.length != arr2.length) return false;

  for (let i = 0; i < arr1.length; i++) {
    if (arr1[i] !== arr2[i]) {
      return false;
    }
  }

  return true;
}

function *seq(start, end, step=1) {
  for (let i=start; i < end; i+=step) {
    yield Math.random()
  }
}
