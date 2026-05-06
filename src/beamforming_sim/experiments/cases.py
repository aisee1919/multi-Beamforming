from __future__ import annotations

from dataclasses import dataclass

from beamforming_sim.scene import AcousticSource, ScanPlane, SourceModel


@dataclass(frozen=True)
class SourceCase:
    """单声源成像工况。

    候选声源集合和“当前只激活一个声源”的实验语义在这里分开。
    """

    index: int
    source: AcousticSource
    plane: ScanPlane

    @property
    def source_model(self) -> SourceModel:
        return SourceModel([self.source])

    @property
    def z_m(self) -> float:
        return float(self.source.position_m[2])

    @property
    def x_m(self) -> float:
        return float(self.source.position_m[0])

    @property
    def y_m(self) -> float:
        return float(self.source.position_m[1])


def build_single_source_cases(source_model: SourceModel, planes: list[ScanPlane]) -> list[SourceCase]:
    """把候选声源列表展开为逐个激活的单声源工况。"""

    plane_by_distance = {plane.distance_m: plane for plane in planes}
    cases: list[SourceCase] = []
    for source_index, source in enumerate(source_model.sources, start=1):
        distance_m = float(source.position_m[2])
        cases.append(SourceCase(index=source_index, source=source, plane=plane_by_distance[distance_m]))
    return cases
