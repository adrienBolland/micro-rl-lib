from system.Hill.Hill import Hill
from system.Maze.Maze import Maze
from system.Maze.MazeSwitches import MazeSwitches

DEFAULT_ENV = {'hill-v0': Hill,
               'maze-v0': Maze,
               'maze-v1': MazeSwitches}


def make(env_name):
    return DEFAULT_ENV[env_name]()


def add_env(env_name, env_class):
    DEFAULT_ENV[env_name] = env_class
