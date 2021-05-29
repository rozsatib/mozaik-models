# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
from model import SelfSustainedPushPull
from experiments import sparse_noise_experiments
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
sparse_noise_data_store = run_experiments(model, sparse_noise_experiments(model), parameters)
sparse_noise_data_store.save()
