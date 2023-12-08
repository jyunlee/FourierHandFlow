import torch
import torch.nn as nn

from .encoders import LBSNet, BaseModule, OccDecoder
from .shapeflow import ShapeFlow


class FWDLBS(LBSNet):
    def __init__(self, num_joints, hidden_size, pn_dim):
        super().__init__(num_joints, hidden_size, pn_dim)

    def get_c_dim(self):
        return self.pn_dim

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['fwd_lbs_config']
        model_state_dict = state_dict['fwd_lbs_model']
        return cls.load(config, model_state_dict)

    @classmethod
    def from_cfg(cls, config):
        model = cls(
            num_joints=config['num_joints'],
            hidden_size=config['hidden_size'],
            pn_dim=config['pn_dim'])

        return model

    def forward(self, points, can_vertices):
        """
        Args:
            points: B x T x 3
            can_vertices: B x N x 3
        Returns:

        """
        vert_code = self.point_encoder(can_vertices)  # B x pn_dim
        point_weights = self._forward(points, vert_code)

        return point_weights  # B x T x K


class ShapeFlowNet(BaseModule):
    def __init__(self,
                 fwd_lbs,
                 leap_occupancy_decoder,
                 shape_net,
                 option: None):
        super(ShapeFlowNet, self).__init__()

        self.fwd_lbs = fwd_lbs
        self.leap_occupancy_decoder = leap_occupancy_decoder
        self.option = option

        self.shape_net = shape_net

    @classmethod
    def from_cfg(cls, config):

        leap_model = cls(
            fwd_lbs=FWDLBS.load_from_file(config['fwd_lbs_model_path']),
            leap_occupancy_decoder=OccDecoder.from_cfg(config),
            shape_net=ShapeFlow.from_cfg(config), 
            option=None
            )

        return leap_model

    @classmethod
    def load_from_file(cls, file_path):
        state_dict = cls.parse_pytorch_file(file_path)
        config = state_dict['leap_model_config']
        model_state_dict = state_dict['leap_model_model']

        leap_model = cls(
            fwd_lbs=FWDLBS.from_cfg(config['fwd_lbs_model_config']),
            leap_occupancy_decoder=OurOccupancyDecoder_StructureOnly_NoCycle.from_cfg(config),
            shape_net=ShapeNet.from_cfg(config), # WARNING!,
            option=None
            )

        leap_model.load_state_dict(model_state_dict)
        return leap_model

    def to(self, **kwargs):
        self.fwd_lbs = self.fwd_lbs.to(**kwargs)
        self.leap_occupancy_decoder = self.leap_occupancy_decoder.to(**kwargs)
        self.shape_net = self.shape_net.to(**kwargs)
        return self

    def eval(self):
        self.fwd_lbs.eval()
        self.leap_occupancy_decoder.eval()
        self.shape_net.eval()
