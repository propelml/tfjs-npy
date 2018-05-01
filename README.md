# NumPy file parsing and serialization for TensorFlow.js

[![Build Status](https://travis-ci.org/propelml/tfjs-npy.svg?branch=master)](https://travis-ci.org/propelml/tfjs-npy)
https://www.npmjs.com/package/tfjs-npy
https://github.com/propelml/tfjs-npy

See https://docs.scipy.org/doc/numpy/neps/npy-format.html for more information
about the file format.

API:

    import { parse, serialize } from "tfjs-npy"

    parse(ab: ArrayBuffer): tf.Tensor

    serialize(tensor: tf.Tensor): Promise<ArrayBuffer>


