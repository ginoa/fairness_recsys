# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:22:39 2019

@author: GinoAlmondo

Info:
Code borrowed from "model.py" from github repo of TCAV, by Google. 
Copyright 2018 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import six
import tensorflow as tf

class ModelWrapper(six.with_metaclass(ABCMeta, object)):
  """Simple wrapper of the for models with session object for TCAV.
    Supports easy inference with no need to deal with the feed_dicts.
  """

  @abstractmethod
  def __init__(self):
    # A dictionary of bottleneck tensors.
    self.bottlenecks_tensors = None
    # A dictionary of input, 'logit' and prediction tensors.
    self.ends = None
    # The model name string.
    self.model_name = None
    # a place holder for index of the neuron/class of interest.
    # usually defined under the graph. For example:
    # with g.as_default():
    #   self.tf.placeholder(tf.int64, shape=[None])
    self.y_input = None
    # The tensor representing the loss (used to calculate derivative).
    self.loss = None

  def _make_gradient_tensors(self):
    """Makes gradient tensors for all bottleneck tensors.
    """
    self.bottlenecks_gradients = {}
    for bn in self.bottlenecks_tensors:
      self.bottlenecks_gradients[bn] = tf.gradients(
          self.loss, self.bottlenecks_tensors[bn])[0]

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: y
    })

  def get_predictions(self, examples):
    """Get prediction of the examples.
    Args:
      imgs: array of examples to get predictions
    Returns:
      array of predictions
    """
    return self.adjust_prediction(
        self.sess.run(self.ends['prediction'], {self.ends['input']: examples}))

  def adjust_prediction(self, pred_t):
    """Adjust the prediction tensor to be the expected shape.
    Defaults to a no-op, but necessary to override for GoogleNet
    Returns:
      pred_t: pred_tensor.
    """
    return pred_t

  def reshape_activations(self, layer_acts):
    """Reshapes layer activations as needed to feed through the model network.
    Override this for models that require reshaping of the activations for use
    in TCAV.
    Args:
      layer_acts: Activations as returned by run_examples.
    Returns:
      Activations in model-dependent form; the default is a squeezed array (i.e.
      at most one dimensions of size 1).
    """
    return np.asarray(layer_acts).squeeze()

  @abstractmethod
  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id)."""
    pass

  @abstractmethod
  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string)."""
    pass

  def run_examples(self, examples, bottleneck_name):
    """Get activations at a bottleneck for provided examples.
    Args:
      examples: example data to feed into network.
      bottleneck_name: string, should be key of self.bottlenecks_tensors
    Returns:
      Activations in the given layer.
    """
    return self.sess.run(self.bottlenecks_tensors[bottleneck_name],
                         {self.ends['input']: examples})
