#!/usr/bin/env python
# coding=utf-8
'''Export keras model as Tensorflow Model for future serving'''

'''Import modules only if necessary. It will help us write codes for torch, CNTK etc.'''
import os
import keras.backend as K

class TFModelExporter():
    def __init__(self):
        return

    def export(self, model, version, path):
        import tensorflow as tf
        K.set_learning_phase(0)
        from tensorflow.python.saved_model import builder 
        from tensorflow.python.saved_model import tag_constants, signature_constants
        from tensorflow.python.saved_model import signature_def_utils_impl
        from tensorflow.python.saved_model.utils import build_tensor_info
        # crate new model: make sure learning_phase=test not train 
        config = model.get_config()
        weights = model.get_weights()
        new_model = Model.from_config(config)
        new_model.set_weights(weights)
        # create model exporter
        export_path = os.path.join(
            tf.compat.as_bytes(export_path), 
            tf.compat.as_bytes(str(version))
            )
        builder = builder.SaveModelBuilder(export_path)
        # create signature
        model_input = build_tensor_info(new_model.input)
        model_output = build_tensor_info(new_model.output)
        predict_signature = signature_def_utils_impl.predict_signature_def(
            inputs={'inputs': model_input}, 
            outputs={'outputs': model_output}
            ) 
        with K.get_session() as sess: # sess is the Tensorflow session that holds your trained model
            builder.add_meta_graph_and_variables(
                sess=sess,             
                tags=[tag_constants.SERVING],
                signature_def_map={
                    'predict_images': predict_signature,
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature
                    },
                legacy_init_op=legacy_init_op
                )
        builder.save()
        return
