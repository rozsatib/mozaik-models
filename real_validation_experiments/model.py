import sys
from parameters import ParameterSet
from mozaik.models import Model
from mozaik.connectors.meta_connectors import GaborConnector
from mozaik.connectors.modular import (
    ModularSamplingProbabilisticConnector,
    ModularSamplingProbabilisticConnectorAnnotationSamplesCount,
)
from mozaik import load_component
from mozaik.space import VisualRegion


class SelfSustainedPushPull(Model):

    required_parameters = ParameterSet(
        {
            "sheets": ParameterSet({"retina_lgn": ParameterSet}),
            "visual_field": ParameterSet,
            "only_afferent": bool,
            "l23": bool,
            "feedback": bool,
            "trial": int,
        }
    )

    def __init__(self, sim, num_threads, parameters):
        Model.__init__(self, sim, num_threads, parameters)

        RetinaLGN = load_component(self.parameters.sheets.retina_lgn.component)

        # Build and instrument the network
        self.visual_field = VisualRegion(
            location_x=self.parameters.visual_field.centre[0],
            location_y=self.parameters.visual_field.centre[1],
            size_x=self.parameters.visual_field.size[0],
            size_y=self.parameters.visual_field.size[1],
        )
        self.input_layer = RetinaLGN(self, self.parameters.sheets.retina_lgn.params)
