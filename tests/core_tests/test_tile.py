from pathml.core.tile import Tile

## pytest.fixture to make a Tile object


def test_tile_repr():
    """bug when i or j = 0, and repr incorrectly shows them as None"""
    tile = tiling.Tile(np.random.randint(0, 255, (20, 20, 3)), i = 0, j = 0)
    assert tile.i == 0
    assert tile.j == 0
    assert repr(tile) == "Tile(array shape (20, 20, 3), i=0, j=0)"
