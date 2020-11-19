# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import json
import mxnet as mx
# Please make sure to import neomxnet
import neomxnet  # noqa: F401
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])
# Change the context to mx.cpu() if deploying to a CPU endpoint
ctx = mx.gpu()

def model_fn(model_dir):
    # The compiled model artifacts are saved with the prefix 'compiled'
    sym, arg_params, aux_params = mx.model.load_checkpoint('compiled', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    exe = mod.bind(for_training=False,
               data_shapes=[('data', (1,3,512,512))],
               label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    # Run warm-up inference on empty data during model load (required for GPU)
    data = mx.nd.empty((1,3,512,512), ctx=ctx)
    mod.forward(Batch([data]))
    return mod


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the GluonNLP model. Called once per request.
    :param model: The Gluon model and the vocab
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
        # pre-processing
    decoded = mx.image.imdecode(data)
    resized = mx.image.resize_short(decoded,512)
    cropped, crop_info = mx.image.center_crop(resized, (512, 512))
    normalized = mx.image.color_normalize(cropped.astype(np.float32) / 255,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))
    
    transposed = normalized.transpose((2, 0, 1))
    batchified = transposed.expand_dims(axis=0)
    casted = batchified.astype(dtype='float32')
    processed_input = casted.as_in_context(ctx)
    
    
    tic = time.time()
    # prediction/inference
    net.forward(Batch([processed_input]))
    # post-processing
    cid, score, bbox = net.get_outputs()
    cid = cid.asnumpy()
    score = score.asnumpy()
    bbox = bbox.asnumpy()
    
    toc = time.time() - tic
    
    response = json.dumps(
        {"class_id":cid.tolist(), 
         "score":score.tolist(), 
         "bbox": bbox.tolist()
        })
                        
    return {"response":response,
            "time": toc
           }


# if __name__=="__main__":
#     net = model_fn(".")
    
#     im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/street_small.jpg?raw=true',
#                           path='street_small.jpg')
    
#     print(transform_fn(net, im_fname, 0, 0))
    
    
    