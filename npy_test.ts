/*!
Copyright 2018 Propel http://propel.site/.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import { test, assertEqual } from "liltest";
import * as tf from "@tensorflow/tfjs-node";
import * as npy from "./npy";
import { readFileSync } from "fs";
const { expectArraysClose } = tf.test_util;

async function load(fn: string): Promise<tf.Tensor> {
  const b = readFileSync(__dirname + "/testdata/" + fn, null);
  const ab = bufferToArrayBuffer(b);
  return await npy.parse(ab);
}

function bufferToArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

test(async function npy_parse() {
  // python -c "import numpy as np; np.save('1.npy', [1.5, 2.5])"
  let t = await load("1.npy");
  assertEqual(t.dataSync(), [1.5, 2.5]);
  assertEqual(t.shape, [2]);
  assertEqual(t.dtype, "float32");

  // python -c "import numpy as np; np.save('2.npy', [[1.5, 43], [13, 2.5]])"
  t = await load("2.npy");
  assertEqual(t.dataSync(), [1.5, 43, 13, 2.5]);
  assertEqual(t.shape, [2, 2]);
  assertEqual(t.dtype, "float32");

  // python -c "import numpy as np; np.save('3.npy', [[[1,2,3],[4,5,6]]])"
  t = await load("3.npy");
  assertEqual(t.dataSync(), [1, 2, 3, 4, 5, 6]);
  assertEqual(t.shape, [1, 2, 3]);
  assertEqual(t.dtype, "int32");

  /*
   python -c "import numpy as np; np.save('4.npy', \
          np.array([0.1, 0.2], 'float32'))"
  */
  t = await load("4.npy");
  expectArraysClose(t.dataSync(), new Float32Array([0.1, 0.2]));
  assertEqual(t.shape, [2]);
  assertEqual(t.dtype, "float32");

  /*
   python -c "import numpy as np; np.save('uint8.npy', \
          np.array([0, 127], 'uint8'))"
  */
  t = await load("uint8.npy");
  expectArraysClose(t.dataSync(), new Int32Array([0, 127]));
  assertEqual(t.shape, [2]);
  assertEqual(t.dtype, "int32"); // TODO uint8
});

test(async function npy_serialize() {
  const t = tf.tensor([1.5, 2.5]);
  const ab = await npy.serialize(t);
  // Now try to parse it.
  const tt = npy.parse(ab);
  expectArraysClose(t.dataSync(), tt.dataSync());
});
