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
import json
import os
import ast
from PIL import Image
import gluoncv
import mxnet as mx
import io
from gluoncv import utils
import time

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    """
    model_name = "faster_rcnn_resnet50_v1b_voc"
    ctx = mx.cpu(0)
    architecure = os.path.join(model_dir, "{}-symbol.json".format(model_name))
    params = os.path.join(model_dir, "{}-0000.params".format(model_name))
    net = mx.gluon.nn.SymbolBlock.imports(architecure, ['data'], params, ctx=ctx)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the GluonNLP model. Called once per request.
    :param model: The Gluon model and the vocab
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    print('Load Images')
    # load images 
    if isinstance(data, str):
        image = ast.literal_eval(data) # For Predictor class
    else:
        image = Image.open(io.BytesIO(data))
        #image = Image.open(data) # For Boto3 Invoke Endpoint
    img = mx.nd.array(image)
                        
    # transform                    
    img = gluoncv.data.transforms.image.imresize(img, 600, 956, interp=9) 
    print(img.shape)
    img = mx.nd.image.to_tensor(img)
    print(img.shape)
    
    nda = img.expand_dims(0)
    print(nda.shape)
#     nda = nda.copyto(gpu(0))
    
    # predictions
    tic = time.time()
    cid, score, bbox = net(nda)    
    cid = cid.asnumpy()
    score = score.asnumpy()
    bbox = bbox.asnumpy()
    toc = time.time() - tic

    # outputs
    response = json.dumps(
        {"class_id":cid.tolist()[0], 
         "score":score.tolist()[0], 
         "bbox": bbox.tolist()[0]
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
    
