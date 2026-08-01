import numpy as np

from main import calculate_spatial_density


def test_detects_crowded_scene():
    pedestrians = [(0, 0, 40, 40), (30, 20, 40, 40), (80, 60, 40, 40)]
    depth_map = np.zeros((200, 200), dtype=np.uint8)
    depth_map[20:60, 20:60] = 40
    depth_map[60:100, 60:100] = 60
    crowded, boxes = calculate_spatial_density(pedestrians, depth_map, crowd_threshold=2, spatial_radius=120, depth_weight=0.3)
    assert crowded is True
    assert len(boxes) >= 1


def test_empty_inputs_do_not_trigger_alert():
    crowded, boxes = calculate_spatial_density([], None)
    assert crowded is False
    assert boxes == []
