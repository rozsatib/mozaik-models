# -*- coding: utf-8 -*-
from model import SelfSustainedPushPull
from experiments import *
import mozaik
from mozaik.controller import prepare_workflow, run_experiments, setup_logging
import mozaik.controller
import sys 
from pyNN import nest

logger = mozaik.getMozaikLogger()

model_class = SelfSustainedPushPull
simulation_name = "SelfSustainedPushPull"

sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
model = model_class(sim,num_threads,parameters)
data_store = run_experiments(model, am_configuration_experiments(model), parameters)
data_store.save()
