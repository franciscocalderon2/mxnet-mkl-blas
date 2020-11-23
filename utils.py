from sagemaker.session import Session
from sagemaker.mxnet.model import MXNetModel
from sagemaker import utils
import numpy as np
import tempfile
import urllib.request


def test_bert(sagemaker_session, ecr_image, instance_type, framework_version, role):
    tmpdir = tempfile.mkdtemp()
    tmpfile = 'bert_sst.tar.gz'
    urllib.request.urlretrieve('https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.tar.gz', tmpfile)

    prefix = 'bert-model'
    model_data = sagemaker_session.upload_data(path=tmpfile, key_prefix=prefix)

    script = "bert_inference.py"
    model = MXNetModel(model_data,
                       role,
                       entry_point=script,
                       source_dir="code",
                       image_uri=ecr_image,
                       py_version="py3",
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('bert')
    print("Deploying...")
    predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)
    
    return predictor

def deploy_bert(model_data, sagemaker_session, ecr_image, instance_type, framework_version, role, script="bert_inference.py"):
    """
    """
    model = MXNetModel(model_data,
                       role,
                       etnry_point=script,
                       source_dir="code",
                       image_uri=ecr_image,
                       py_version="py3",
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('bert')
    print("Deploying...")
    predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)
    
    print("\nEndpoint Name: {}".format(endpoint_name))
    
    return predictor

def deploy_model(model_data, endpoint_name, sagemaker_session, ecr_image, instance_type, framework_version, role, script="bert_inference.py", source_dir="code/"):
    """
    """
    if ecr_image is not None:
        model = MXNetModel(model_data,
                           role,
                           entry_point=script,
                           source_dir=source_dir,
                           image_uri=ecr_image,
                           py_version="py3",
                           framework_version=framework_version,
                           sagemaker_session=sagemaker_session)
    else:
        model = MXNetModel(model_data,
                           role,
                           entry_point=script,
                           source_dir=source_dir,
                           py_version="py3",
                           framework_version=framework_version,
                           sagemaker_session=sagemaker_session)
        

    endpoint_name = utils.unique_name_from_base(endpoint_name)
    print("Deploying...")
    predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)
    
    print("\nEndpoint Name: {}".format(endpoint_name))
    
    return predictor


def inference(data, predictor, loops=10):
    times = []
    for i in range(loops):
        output = predictor.predict(data)
        times.append(output['time'])
    
    print("Avg:", np.mean(times), "Std:", np.std(times), "with {} loops".format(loops))
    return times