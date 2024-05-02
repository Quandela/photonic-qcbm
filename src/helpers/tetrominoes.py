import numpy as np
from numpy.random import randint
from typing import Tuple


class Tetrominoes:
    """
    Tetrominoes class used to generate Tetris geometric shapes on a grid.

    :param height_grid: Height of the grid where the tetrominoes will be generated.
    :param width_grid: Width of the grid where the tetrominoes will be generated.
    :param flat_grid: If True, return generated data as a flattened 1d vector.
    """
    def __init__(self, height_grid: int = 4, width_grid: int = 4, flat_grid: bool = True):
        self._height_grid = height_grid
        self._width_grid = width_grid
        self._shape_list = ['L', 'O', 'T', 'I', 'S', 'J', 'Z']
        self._direction_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._flat_grid = flat_grid
    
    @staticmethod
    def turn_right(direction: Tuple):
        vertical = direction[0]
        horizontal = direction[1]
        return horizontal, -vertical

    @staticmethod
    def turn_left(direction: Tuple):
        vertical = direction[0]
        horizontal = direction[1]
        return -horizontal, vertical
    
    def generate_random_setting(self):
        starting_point = (randint(0, self._height_grid - 1),
                          randint(0, self._width_grid - 1))
        shape = self._shape_list[randint(0, len(self._shape_list))]
        direction = self._direction_list[randint(0, len(self._direction_list))]
        return starting_point, shape, direction

    def generate_samples(self, n_samples):
        i = 0
        dataset = []
        shape_record = []
        while i < n_samples:
            starting_point, shape, direction = self.generate_random_setting()
            try:
                grid = self.create_tetromino(starting_point, shape, direction)
                i += 1
                dataset.append(grid)
                shape_record.append(shape)
            except:
                continue
        
        return dataset, shape_record
    
    def create_tetromino(self, starting_point, shape, direction):
        # blank grid
        grid = np.zeros((self._height_grid, self._width_grid))
        
        # tile 1
        grid[starting_point] = 1
        
        # tile 2
        tile_2 = tuple(np.add(starting_point, direction))

        if all(element >= 0 for element in tile_2):
            grid[tile_2] = 1
        else:
            raise ValueError('Tile 2 is outside of grid')
            
        # tile 3
        if shape in ['I', 'J', 'L', 'T']:
            tile_3 = tuple(np.add(tile_2, direction))
        elif shape in ['O', 'Z']:
            tile_3 = tuple(np.add(tile_2, self.turn_right(direction)))
        elif shape in ['S']:
            tile_3 = tuple(np.add(tile_2, self.turn_left(direction)))
        else:
            raise ValueError(f"Unexpected shape {shape} for tile 3")

        if all(element >= 0 for element in tile_3):
            grid[tile_3] = 1
        else:
            raise ValueError('Tile 3 is outside of grid')
            
        # tile 4
        if shape in ['I', 'S', 'Z']:
            tile_4 = tuple(np.add(tile_3, direction))
        elif shape in ['O']:
            tile_4 = tuple(np.subtract(tile_3, direction))
        elif shape in ['L']:
            tile_4 = tuple(np.add(tile_3, self.turn_right(direction)))
        elif shape in ['J']:
            tile_4 = tuple(np.add(tile_3, self.turn_left(direction)))
        elif shape in ['T']:
            tile_4 = tuple(np.add(tile_2, self.turn_right(direction)))
        else:
            raise ValueError(f"Unexpected shape {shape} for tile 4")

        if all(element >= 0 for element in tile_4):
            grid[tile_4] = 1
        else:
            raise ValueError('Tile 4 is outside of grid')

        return grid.flatten() if self._flat_grid else grid
