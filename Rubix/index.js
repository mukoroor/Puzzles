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

async function runMatrixMultiplyWithStats(size, warmupRuns) {
  if (!navigator.gpu) {
      console.error("WebGPU not supported on this browser.");
      return;
  }

  // Request WebGPU Adapter & Device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({
    // requiredFeatures: ["subgroups"],
  });

  // Generate random matrices
  const A = Tensor.from([...seq(0, size * size)], [size, size], Float32Array);
  const B = Tensor.from([...seq(0, size * size)], [size, size], Float32Array);

  const A_gpu = await Tensor.sendToGPUDevice(A, device);
  const B_gpu = await Tensor.sendToGPUDevice(B, device);

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
