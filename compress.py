#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

class CompressionNet:
    def __init__(self, target_dim=512):
        if target_dim == 512:
            self.saved_model_dir = "network/compression_network_768_to_512_305284epochs_3Layer512.pb"
            self.max_file = "network/compression_network_768_to_512_305284epochs_3Layer512_max.npy"
            self.min_file = "network/compression_network_768_to_512_305284epochs_3Layer512_min.npy"
            self.output_tensor_name = "a3/BiasAdd:0"
        elif target_dim == 64:
            self.saved_model_dir = "network/compression_network_768_to_64_468004epochs_3Layer512.pb"
            self.max_file = "network/compression_network_768_to_64_468004epochs_3Layer512_max.npy"
            self.min_file = "network/compression_network_768_to_64_468004epochs_3Layer512_min.npy"
            self.output_tensor_name = "a3/BiasAdd:0"
        else:
            raise Exception("No compression configuration found for target dimension", target_dim)

        self.target_dim = target_dim
        
        self.max_values = np.max(np.load(self.max_file))
        self.min_values = np.min(np.load(self.min_file))
        self.diff_values = self.max_values - self.min_values
        
        # enable TensorFlow 1.x compatibility mode
        tf.compat.v1.disable_eager_execution()
        
        # create computation graph and run session 
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session(graph=self.graph)
            
            # load existing TensorFlow 1.x model
            tf.compat.v1.saved_model.loader.load(self.session, [tf.saved_model.SERVING], self.saved_model_dir)
            
            # extract the input and output tensor
            self.input_tensor = self.graph.get_tensor_by_name("cv_data:0")
            self.output_tensor = self.graph.get_tensor_by_name(self.output_tensor_name)

        # dry run
        sample_size = 10000
        sample_data = np.zeros((sample_size, self.input_tensor.shape[1]), dtype=np.float32)
        output = self.compress(sample_data, quantize=False, batch_size=sample_size)
        print('Loaded and tested the compression network with shape {}'.format(sample_data.shape))

            
    def compress(self, input_data, quantize=False, batch_size=8192):        
        output = []
        with self.graph.as_default():
            for input_data_batch in self.batch_generator_(input_data, batch_size=batch_size):
                trans_fv = self.session.run(
                    self.output_tensor, feed_dict={self.input_tensor: input_data_batch.astype(np.float32)}
                )
                quant_output = self.quantize_to_uint8_(trans_fv)
                if not quantize:
                    quant_output = quant_output.astype(np.float32)
                output.extend(quant_output)
        output = np.stack(output)
        
        return output

    @staticmethod
    def batch_generator_(vectors, batch_size=64):
        for start_idx in range(0, vectors.shape[0], batch_size):
            if (start_idx + batch_size) < vectors.shape[0]:
                excerpt = slice(start_idx, start_idx + batch_size)
            else:
                excerpt = slice(start_idx, vectors.shape[0])
            yield vectors[excerpt]
            
    def quantize_to_uint8_(self, input_data):     
        normalized_data = np.clip((input_data - self.min_values) / self.diff_values, 0, 1)
        return (normalized_data * 255).astype(np.uint8)
