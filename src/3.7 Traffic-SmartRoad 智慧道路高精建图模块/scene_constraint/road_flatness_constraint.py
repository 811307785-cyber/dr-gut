"""
道路平整度约束模块（场景专属）
底层依赖：Common-LSG constraint_algo 基类
功能：约束道路平整度、坡度，提取交通设施层级
"""
from Common_LSG.constraint_algo import BaseSectionConstraint

class RoadFlatnessConstraint(BaseSectionConstraint):
    def road_surface_constrain(self, section_matrix, max_slope=0.05):
        """道路平整度约束，校正路面微小起伏伪影"""
        constrained_matrix = self.flatness_constrain(section_matrix, max_slope=max_slope)
        return constrained_matrix

    def traffic_facility_extract(self, section_stack, facility_type="sign"):
        """交通设施层级提取，分离标牌、护栏、路灯等结构"""
        facility_mask = self.height_layer_segment(section_stack, facility_type=facility_type)
        return facility_mask