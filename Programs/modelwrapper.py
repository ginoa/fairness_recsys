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

import tcav
import numpy as np

class myModelWrapper(tcav.model.ModelWrapper):
  # custom class for overriding abstract methods of tcav.model.ModelWrapper
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

  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id)."""
    pass

  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string)."""
    pass