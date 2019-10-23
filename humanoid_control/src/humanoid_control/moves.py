import csv
from os import path
import rospkg
import rospy


rospack = rospkg.RosPack()
moves_dir = path.join(rospack.get_path('humanoid_control'), 'moves')


def get_path_to_move(name):
  return path.join(moves_dir, name + '.csv')


def load_move(path):
  try:
    with open(path, newline='') as file:
      return list(csv.reader(file, delimiter=','))
  except:
    rospy.logerr('Não foi possível carregar o arquivo ' + path + '!')
    return []


def get_move_generator(source):
		if isinstance(source, str):
			with open(source, newline='') as file:
				table = csv.reader(file, delimiter=',')
				for state in table:
					yield state
		else:
			for state in source:
				yield state
