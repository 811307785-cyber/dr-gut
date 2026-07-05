"""
地震勘探噪声压制模块（场景专属）
底层依赖：Common-LSG constraint_algo 平滑算子
功能：压制面波、多次波等地震干扰波，提升地层信噪比
"""
from Common_LSG.constraint_algo import AdaptiveSmoother

class SeismicNoiseSuppress(AdaptiveSmoother):
    def surface_wave_remove(self, seismic_section):
        """面波干扰压制"""
        cleaned_section = self.frequency_filter(seismic_section, low_cut=5, high_cut=60)
        return cleaned_section

    def multiple_wave_remove(self, seismic_stack):
        """多次波伪影消除"""
        cleaned_stack = self.periodic_reflection_filter(seismic_stack)
        return cleaned_stack