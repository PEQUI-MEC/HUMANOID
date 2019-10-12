# -*- coding: utf-8 -*-
import csv
from functools import reduce
import math
import numpy as np
import rospy
import threading
import time

from .body_physics import BodyPhysics
from .utils import sigmoid_deslocada

KP_CONST = 0.3

DEG_TO_RAD = math.pi * 180

class Control():
	def __init__(self,
              altura_inicial=17.,
              tempo_passo=0.3,
              deslocamento_ypelves=2.,
              deslocamento_zpes=3.,
              deslocamento_xpes=1.5,
              deslocamento_zpelves=30.,
              gravity_compensation_enable=False):
		self.state = 'IDLE'
		self.enable = False
		self.manual_mode = True

		self.altura = altura_inicial
		self.pos_inicial_pelves = [0., 1.4, altura_inicial]
		self.pos_inicial_foot = [0., 1.4, altura_inicial]
		self.deslocamentoXpes = 0.
		self.deslocamentoYpelves = 0
		self.deslocamentoZpes = 0
		self.deslocamentoZpelves = 0
		self.deslocamentoXpesMAX = deslocamento_xpes
		self.deslocamentoZpesMAX = deslocamento_zpes
		self.deslocamentoYpelvesMAX = deslocamento_ypelves
		self.deslocamentoZpelvesMAX = deslocamento_zpelves

		self.torsoOffsetMin = 5 * math.pi/180.
		self.torsoOffsetMax = 13 * math.pi/180.

		self.nEstados = 125
		self.tempoPasso = tempo_passo
		self.a = 10.3
		self.c = 9.5

		self.angulos = np.zeros(18)
		self.fps_count = 0
		self.last_time = 0
		self.count_frames = 0
		self.timer_fps = 0
		self.deltaTime = 0
		self.time_ignore_GC = 0.1 #entre 0 e 1 - porcentagem de tempo para ignorar o gravity compensation

		# TODO: Usar o rospy.Rate para controle do loop
		self.simTransRate = 1/self.nEstados*self.tempoPasso

		self.tempo_acelerando = 4.
		self.tempo_marchando = 4.
		self.tempo_virando = 3.

		self.visao_search = False
		self.visao_bola = False
		self.turn90 = False
		self.max_yaw = 20
		self.min_yaw = 5

		self.robo_roll = 0
		self.robo_yaw = 0
		self.robo_pitch = 0

		self.gimbal_yaw = 0
		self.gimbal_pitch = 0

		self.robo_yaw_lock = 0
		self.robo_pitch_lock = 0
		self.gravity_compensation_enable = gravity_compensation_enable
		self.Lfoot_press = [0,0,0,0]
		self.Rfoot_press = [0,0,0,0]
		self.total_press = 0

		self.t_state = 0
		self.rot_desvio = 0
		self.rota_dir = 0
		self.rota_esq = 0
		self.angulo_vira = 3

		self.marchando = False
		self.recuando = False
		self.acelerando = False
		self.freando = False
		self.ladeando = False
		self.desladeando = False
		self.interpolando = False
		self.posicionando = False

		# Perna no chão: 1 = direita; 0 = esquerda
		self.perna = 0

		self.body = BodyPhysics()
		self.RIGHT_ANKLE_ROLL = 0
		self.RIGHT_ANKLE_PITCH = 1
		self.RIGHT_KNEE = 2
		self.RIGHT_HIP_PITCH = 3
		self.RIGHT_HIP_ROLL = 4
		self.RIGHT_HIP_YALL = 5
		self.LEFT_ANKLE_ROLL = 6
		self.LEFT_ANKLE_PITCH = 7
		self.LEFT_KNEE = 8
		self.LEFT_HIP_PITCH = 9
		self.LEFT_HIP_ROLL = 10
		self.LEFT_HIP_YALL = 11
		self.LEFT_ARM_PITCH = 12
		self.LEFT_ARM_YALL = 13
		self.LEFT_ARM_ROLL = 14
		self.RIGHT_ARM_PITCH = 15
		self.RIGHT_ARM_YALL = 16
		self.RIGHT_ARM_ROLL = 17

		self.state_encoder = {
			"IDLE" : 1,
			"MARCH" : 2,
			"WALK"  : 3,
			"TURN"  : 4,
			"FALLEN": 5,
			"UP"    : 6,
			"PENALIZED": 7,
			"TURN90": 8
		}

		try:
			with open ('estados_levanta_frente.csv', newline='') as csvfile:
				tabela = list(csv.reader(csvfile, delimiter=','))
				tabela = np.array(tabela)
				self.estados_levanta_frente = tabela[1:, :]
				self.tempos_levanta_frente = [4]*19

			with open('estados_levanta_back.csv', newline='') as csvfile2:
				tabela = list(csv.reader(csvfile2, delimiter=','))
				tabela = np.array(tabela)
				self.estados_levanta_costas = tabela[1:, :]
				self.tempos_levanta_costas = [4]*19
		except Exception:
			rospy.logerr('Não foi possível carregar os arquivos CSV')
			self.estados_levanta_costas = []
			self.tempos_levanta_costas = []
			self.estados_levanta_frente = []
			self.tempos_levanta_frente = []

		self.running = True


	def reset(self):
		self.enable = False
		self.state = 'IDLE'
		self.visao_bola = False


	def atualiza_fps(self):
		if self.timer_fps >= 1:
			self.fps_count = self.count_frames
			self.count_frames = 0
			self.timer_fps = 0
			return self.fps_count
		self.deltaTime = time.time() - self.last_time
		self.last_time = time.time()
		self.count_frames += 1
		self.timer_fps += self.deltaTime
		return None


	# '''
	# 	- descrição: função que recebe informações de onde está a bola,
	#     atualizando as variaveis globais referêntes ao gimbal
	#
	# 	- entrada: vetor "data" de 3 posições (sugeito a modificações, dependendo da lógica da visão)
	# 		data[0] = posição angular da bola no eixo pitch (y)
	# 		data[1] = posição angular da bola no eixo yall (z)
	# 		data[2] = flag que indica se está com a bola, usada para setar o
	#       estado do controle para IDLE ou permitir que o robô ande
	# '''
	def vision_status_callback(self, msg):
		if self.manual_mode:
			return

		(pitch, yaw, bola) = msg.data
		if self.robo_yaw + yaw < 0:
			self.gimbal_yaw = self.robo_yaw + yaw + 360
		elif self.robo_yaw + yaw > 360:
			self.gimbal_yaw = (self.robo_yaw + yaw)% 360
		else:
			self.gimbal_yaw = self.robo_yaw + yaw

		self.gimbal_pitch = pitch
		self.visao_bola = bool(bola)


	# 	'''
	# 		- descrição: função que recebe dados do sensor de pressão dos pés e
	#         atualiza as variaveis globais correspondentes.
	# 		- entrada: vetor "data" de 8 posições:
	# 			data [1:4] = valores [p1,p2,p3,p4] que indicam o nivel de força
	#           detectados nos pontos na extremidade do pé esquerdo
	# 			data [4:8] = valores [p1,p2,p3,p4] que indicam o nivel de força
	#           detectados nos pontos na extremidade do pé direito
	# 	'''
	# 	Leitura sensores de pressão
	def foot_pressure_callback(self, msg):
		self.Lfoot_press = msg.data[:4]
		self.Rfoot_press = msg.data[4:]
		self.total_press = np.sum(self.Lfoot_press)+np.sum(self.Rfoot_press)


	# 	Leitura IMU - robo
	def robot_inertial_callback(self, msg):
		self.robo_yaw = msg.data[2]
		self.robo_pitch = msg.data[1]
		self.robo_roll = msg.data[0]

		if (abs(self.robo_pitch) > 45 or abs(self.robo_roll) > 45) and not self.interpolando:
			self.state = 'FALLEN'


	def classifica_estado(self):
		if self.state is 'IDLE':
			if not self.enable:
				return -1
			if self.turn90:
				return 'MARCH'
			elif self.visao_bola:
				return 'MARCH'
			else:
				return -1
		elif self.state is 'TURN90':
			if abs(self.robo_yaw_lock) <= self.min_yaw:
				return 'MARCH'
			else:
				return -1
		elif self.state is 'MARCH':
			if self.turn90:
				return 'TURN90'
			elif not self.visao_bola:
				return 'IDLE'
			elif self.visao_bola and abs(self.robo_yaw_lock) > self.max_yaw:
				return 'TURN'
			elif self.visao_bola and self.robo_pitch_lock > -45:
				return 'WALK'
			else:
				return -1
		elif self.state is 'WALK':
			if not self.visao_bola or abs(self.robo_yaw_lock) > self.max_yaw or self.robo_pitch_lock <= -45:
				return 'MARCH'
			else:
				return -1
		elif self.state is 'TURN':
			if not self.visao_bola or abs(self.robo_yaw_lock) < self.min_yaw:
				return 'MARCH'
			else:
				return -1
		else:
			rospy.logerr('O estado "{}" não existe'.format(self.state))


	def gravity_compensation(self):
		if not self.gravity_compensation_enable or (self.t_state < self.tempoPasso/2 and self.t_state < self.tempoPasso*self.time_ignore_GC) or (self.t_state >= self.tempoPasso/2 and self.t_state > self.tempoPasso*(1-self.time_ignore_GC)) or self.deslocamentoYpelves != self.deslocamentoYpelvesMAX or self.state is "IDLE":
			return

		torques = self.body.get_torque_in_joint(self.perna,[3,5])
		dQ = (np.array(torques)/KP_CONST)/15
		dQ *= math.sin(self.t_state*math.pi/self.tempoPasso)

		if self.perna:
			self.angulos[self.RIGHT_KNEE] += dQ[0]
			self.angulos[self.RIGHT_HIP_ROLL] += (dQ[1]*-1)
		else:
			self.angulos[self.LEFT_KNEE] += dQ[0]
			self.angulos[self.LEFT_HIP_ROLL] += dQ[1]


	def posiciona_robo(self):
		if self.robo_yaw > self.gimbal_yaw:
			esq_angle = self.robo_yaw - self.gimbal_yaw
			dir_angle = 360 - esq_angle
		else:
			dir_angle = self.gimbal_yaw - self.robo_yaw
			esq_angle = 360 - dir_angle
		# dir_angle =   angulo a virar para a direita  para ajustar ao gimbal
		# esq_angle = - angulo a virar para a esquerda para ajustar ao gimbal
		if esq_angle > dir_angle:
			self.robo_yaw_lock = dir_angle
		else:
			self.robo_yaw_lock = -esq_angle

		self.robo_pitch_lock = self.gimbal_pitch


	def run(self):
		timer_main_loop = 0
		self.perna = 0 # perna direita(1) ou esquerda(0) no chão

		while (self.running):
			if (self.state is 'FALLEN'):
				if not self.interpolando:
					self.levanta()
			elif self.state is 'MARCH':
				if self.deslocamentoYpelves != self.deslocamentoYpelvesMAX:
					self.marchar()
				elif self.deslocamentoXpes != 0:
					self.freia_frente()
				else:
					novo_estado = self.classifica_estado()
					if novo_estado != -1:
						self.state = novo_estado
			elif self.state is 'IDLE':
				if self.rota_dir != 0 or self.rota_esq != 0:
					self.para_de_virar()
				elif self.deslocamentoXpes != 0:
					self.freia_frente()
				elif self.deslocamentoYpelves != 0:
					self.recuar()
				else:
					novo_estado = self.classifica_estado()
					if novo_estado != -1:
						self.state = novo_estado
			elif self.state is 'WALK':
				if self.deslocamentoXpes < self.deslocamentoXpesMAX:
					self.acelera_frente()
				else:
					novo_estado = self.classifica_estado()
					if novo_estado != -1:
						self.state = novo_estado
			elif self.state is 'TURN':
				if abs(self.robo_yaw_lock) > self.min_yaw:
					self.vira()
				elif self.rota_dir != 0 or self.rota_esq != 0:
					self.para_de_virar()
				else:
					novo_estado = self.classifica_estado()
					if novo_estado != -1:
						self.state = novo_estado
			elif self.state is 'TURN90':
				if self.deslocamentoYpelves < self.deslocamentoYpelvesMAX:
					self.marchar()
				elif abs(self.robo_yaw_lock) > self.min_yaw:
					self.vira()
				elif self.rota_dir != 0 or self.rota_esq != 0:
					self.para_de_virar()
				else:
					novo_estado = self.classifica_estado()
					if novo_estado != -1:
						self.turn90 = False
						self.state = novo_estado
			elif self.state is 'UP':
				if self.rota_dir != 0 or self.rota_esq != 0:
					self.para_de_virar()
				elif self.deslocamentoXpes != 0:
					self.freia_frente()
				elif self.deslocamentoYpelves != 0:
					self.recuar()
				else:
					#robo pronto para levantar
					pass
			elif self.state is 'PENALIZED':
				if self.rota_dir != 0 or self.rota_esq != 0:
					self.para_de_virar()
				elif self.deslocamentoXpes != 0:
					self.freia_frente()
				elif self.deslocamentoYpelves != 0:
					self.recuar()

			self.atualiza_fps()
			self.atualiza_estado()
			self.atualiza_cinematica()
			self.gravity_compensation()
			self.posiciona_robo()
			timer_main_loop += self.deltaTime
			time.sleep(self.simTransRate)


	# Anda de lado para alinhar com o gol
	def posiciona(self):
		if not self.posicionando:
			self.posicionando = True
			self.timer_reposiciona = 0
		if self.posicionando:
			self.timer_reposiciona += self.deltaTime
			if self.robo_yaw > 270:
				#anda de lado para a esquerda
				if self.perna:
					self.anda_de_lado_esquerda()
				else:
					self.desanda_de_lado_esquerda()
			elif self.robo_yaw < 90:
				#anda de lado para a direita
				if not self.perna:
					self.anda_de_lado_direita()
				else:
					self.desanda_de_lado_direita()
		if self.timer_reposiciona > self.tempoPasso*6:
			if abs(self.pos_inicial_pelves[1]) < 0.01:
				self.pos_inicial_pelves[1] = 0
				self.posicionando = False
				self.state = 'IDLE'
			else:
				if self.pos_inicial_pelves[1] > 0 and not self.perna:
					self.desanda_de_lado_esquerda()
				if self.pos_inicial_pelves[1] < 0 and self.perna:
					self.desanda_de_lado_direita()


	def levanta(self):
		if not self.interpolando:
			self.interpolando = True
			if (self.robo_pitch >0):
				t = threading.Thread(target=self.interpola_estados, args=[self.estados_levanta_frente, self.tempos_levanta_frente])
				t.daemon = True
				t.start()
			else:
				t = threading.Thread(target=self.interpola_estados, args=[self.estados_levanta_costas, self.tempos_levanta_costas])
				t.daemon = True
				t.start()


	def anda_de_lado_esquerda(self):
		if (not self.ladeando or self.desladeando)and self.pos_inicial_pelves[1] != self.deslocamentoYpelvesMAX/8:
			self.ladeando = True
			self.desladeando = False
			self.timer_movimentacao = 0
		if self.pos_inicial_pelves[1] != self.deslocamentoYpelvesMAX/8:
			self.timer_movimentacao += self.deltaTime
			self.pos_inicial_pelves[1] = sigmoid_deslocada(self.timer_movimentacao, self.tempoPasso)*self.deslocamentoYpelvesMAX/8
		if abs (self.pos_inicial_pelves[1] - self.deslocamentoYpelvesMAX/8) <= 0.01:
			self.pos_inicial_pelves[1] = self.deslocamentoYpelvesMAX/8
			self.ladeando = False


	def desanda_de_lado_esquerda(self):
		if (not self.desladeando or self.ladeando) and self.pos_inicial_pelves[1] != 0.:
			self.desladeando = True
			self.ladeando = False
			self.timer_movimentacao = 0
		if self.pos_inicial_pelves[1] != 0.:
			self.timer_movimentacao += self.deltaTime
			self.pos_inicial_pelves[1] = (1-sigmoid_deslocada(self.timer_movimentacao, self.tempoPasso))*self.deslocamentoYpelvesMAX/8
		if self.pos_inicial_pelves[1] <= 0.01:
			self.pos_inicial_pelves[1] = 0.
			self.desladeando = False


	def anda_de_lado_direita(self):
		if (not self.ladeando or self.desladeando)and self.pos_inicial_pelves[1] != self.deslocamentoYpelvesMAX/8:
			self.ladeando = True
			self.desladeando = False
			self.timer_movimentacao = 0
		if self.pos_inicial_pelves[1] != self.deslocamentoYpelvesMAX/8:
			self.timer_movimentacao += self.deltaTime
			self.pos_inicial_pelves[1] = -sigmoid_deslocada(self.timer_movimentacao, self.tempoPasso)*self.deslocamentoYpelvesMAX/8
		if abs (self.pos_inicial_pelves[1] - self.deslocamentoYpelvesMAX/8) <= 0.01:
			self.pos_inicial_pelves[1] = self.deslocamentoYpelvesMAX/8
			self.ladeando = False


	def desanda_de_lado_direita(self):
		if (not self.desladeando or self.ladeando) and self.pos_inicial_pelves[1] != 0.:
			self.desladeando = True
			self.ladeando = False
			self.timer_movimentacao = 0
		if self.pos_inicial_pelves[1] != 0.:
			self.timer_movimentacao += self.deltaTime
			self.pos_inicial_pelves[1] = (-1 +sigmoid_deslocada(self.timer_movimentacao, self.tempoPasso))*self.deslocamentoYpelvesMAX/8
		if self.pos_inicial_pelves[1] <= 0.01:
			self.pos_inicial_pelves[1] = 0.
			self.desladeando = False


	# 	'''
	# 		- Define para qual lado o robô deve virar com base no yall lock
	# 	'''
	def vira(self):
		if self.robo_yaw_lock < 0:
			self.rot_desvio = 1
		else:
			self.rot_desvio = -1


	# 	'''
	# 		- Vai parando de virar pelo tempo definido no construtor
	# 	'''
	def para_de_virar(self):
		self.rot_desvio = 0


	# 	'''
	# 		- Interpola distância de deslocamento dos pés, da atual até o max setado no contrutor
	# 	'''
	def acelera_frente(self):
		if not self.acelerando and self.deslocamentoXpes != self.deslocamentoXpesMAX:
			self.acelerando = True
			self.timer_movimentacao = 0
		if self.deslocamentoXpes != self.deslocamentoXpesMAX:
			self.timer_movimentacao += self.deltaTime
			self.deslocamentoXpes = sigmoid_deslocada(self.timer_movimentacao, self.tempo_acelerando)*self.deslocamentoXpesMAX
		if abs(self.deslocamentoXpes - self.deslocamentoXpesMAX) <= 0.01:
			self.deslocamentoXpes = self.deslocamentoXpesMAX
			self.acelerando = False


	# 	'''
	# 		- Interpola distância de deslocamento dos pés, diminuindo este valor até que se torne 0
	# 	'''
	def freia_frente(self):
		if not self.freando and self.deslocamentoXpes != 0:
			self.freando = True
			self.timer_movimentacao = 0
		if self.deslocamentoXpes != 0:
			self.timer_movimentacao += self.deltaTime
			self.deslocamentoXpes = (1. - sigmoid_deslocada(self.timer_movimentacao, self.tempo_acelerando))*self.deslocamentoXpesMAX
		if self.deslocamentoXpes  <= 0.01:
			self.deslocamentoXpes = 0
			self.freando = False


	# 	'''
	# 		- Interpola deslocamento lateral da pelves e o deslocamento para cima dos pés, da atual até o max
	# 	'''
	def marchar(self):
		if (not self.marchando) and self.deslocamentoYpelves != self.deslocamentoYpelvesMAX:
			self.marchando = True
			self.timer_movimentacao = 0
		if self.deslocamentoYpelves != self.deslocamentoYpelvesMAX:
			self.timer_movimentacao += self.deltaTime
			self.deslocamentoZpes = sigmoid_deslocada(self.timer_movimentacao, self.tempo_marchando)*self.deslocamentoZpesMAX
			self.deslocamentoYpelves = sigmoid_deslocada(self.timer_movimentacao, self.tempo_marchando)*self.deslocamentoYpelvesMAX
		if abs(self.deslocamentoYpelves - self.deslocamentoYpelvesMAX) <= 0.01:
			self.deslocamentoZpes = self.deslocamentoZpesMAX
			self.deslocamentoYpelves = self.deslocamentoYpelvesMAX
			self.marchando = False


	# 	'''
	# 		- Interpola deslocamento lateral da pelves e o deslocamento para cima dos pés,
	#           diminuindo estes valores até chegar em 0
	# 	'''
	def recuar(self):
		if not self.recuando and self.deslocamentoYpelves != 0:
			self.recuando = True
			self.timer_movimentacao = 0
		if self.deslocamentoYpelves != 0:
			self.timer_movimentacao += self.deltaTime
			self.deslocamentoZpes = (1. - sigmoid_deslocada(self.timer_movimentacao, self.tempo_marchando))*self.deslocamentoZpesMAX
			self.deslocamentoYpelves = (1. - sigmoid_deslocada(self.timer_movimentacao, self.tempo_marchando))*self.deslocamentoYpelvesMAX
		if self.deslocamentoYpelves <= 0.01:
			self.deslocamentoZpes = 0
			self.deslocamentoYpelves = 0
			self.recuando = False


	#Change state
	def atualiza_estado(self):
		# incrementa currentStateTime até tempoPasso (até trocar voltar à fase de suporte duplo)
		self.t_state += self.deltaTime
		if self.t_state >= self.tempoPasso:
			self.t_state = 0
			# indica se é a perna direita (1) ou esquerda(0) no chão
			self.perna = (self.perna+1)%2
			if self.rot_desvio != 0:
				if self.rot_desvio > 0:
					if self.perna:
						self.rota_dir = -1
						self.rota_esq *= 2
					else:
						self.rota_esq = -1
						self.rota_dir *= 2
				else:
					if self.perna:
						self.rota_dir = 1
						self.rota_esq *= 2
					else:
						self.rota_esq = 1
						self.rota_dir *= 2
			else:
				if math.fabs(self.rota_esq) == 2:
					self.rota_esq = 0
				elif math.fabs(self.rota_esq) == 1:
					self.rota_esq *= 2
				if math.fabs(self.rota_dir) == 2:
					self.rota_dir = 0
				elif math.fabs(self.rota_dir) == 1:
					self.rota_dir *= 2


# 	'''
# 		- Retorna os 6 angulos de da perna, calculando a cinematica inversa.
#           Considerando o pé como base e o quadril como ponto variável
# 	'''
	def footToHip(self, pointHip):
		angulos = []
		x,y,z = pointHip

		#ankle roll
		theta = math.atan(y/z)
		angulos.append(theta)

		#ankle pitch
		b = math.sqrt(x**2+y**2+z**2)
		a_2 = self.a**2
		b_2 = b**2
		c_2 = self.c**2
		anguloA = math.acos((a_2-(b_2+c_2))/(-2*b*self.c))
		betha = math.atan(x/z)
		anguloA = betha + anguloA
		angulos.append(anguloA)

		#knee
		anguloB = math.acos((b_2-(a_2+c_2))/(-2*self.a*self.c))
		anguloB = anguloB - math.pi
		angulos.append(anguloB)

		#hip pitch
		anguloC = math.acos((c_2-(a_2+b_2))/(-2*self.a*b))
		anguloC = anguloC - betha
		angulos.append(anguloC)

		#hip roll
		angulos.append(theta)

		#hip yall
		angulos.append(0)

		return angulos


# 	'''
# 		- Pega o proximo "estado" da função de trajetória, a função de trajetória muda
#         de acordo com as variaveis que definem o deslocamento e rotação do robô

# 		Entrada: tempo float/int t
# 		Saída: 2 vetores de 3 posições (x,y,z). O primeiro indica a posição da pelves
#              considerando o pé em contato com o chão como base,
# 			   o segundo vetor indica a posição do pé de balanço considerando a pelves do pé de balanço como base.
# 	'''
	def getTragectoryPoint(self, x):
		pos_pelves = self.pos_inicial_pelves[:]

		# nEstados * [-0.5,0.5]
		# aux_estados = (x-self.N_ESTADOS/2)

		# deslocamentoXpes/2 * tgh(x)

		dif_estado = (x-self.nEstados/2)

		aux = (2*dif_estado)/50
		aux2 = ((math.exp(aux) - math.exp(- aux))/(math.exp(aux)+math.exp(-aux)))

		p1 = (self.deslocamentoXpes/2)*aux2
		pos_pelves[0] = p1 + 1.8
		pos_pelves[1] += -self.deslocamentoYpelves*math.sin(x*math.pi/self.nEstados)

		pos_foot = self.pos_inicial_pelves[:]
		p2 = (-self.deslocamentoXpes/2)*aux2
		pos_foot[0] = p2 + 1.8
		pos_foot[1] += self.deslocamentoYpelves*math.sin(x*math.pi/self.nEstados)
		pos_foot[2] = self.altura - self.deslocamentoZpes*math.exp(-(dif_estado**2)/600)
		return pos_pelves, pos_foot


	# interpolação simples entre estados
	def interpola_estados(self, estados, tempos):
		if len(estados) > 0:
			p_ant = estados[0]
		else:
			return
		for i in range(1, len(estados)):
			p_atual = estados[i]
			t = tempos[i-1]
			timer = 0
			m = []
			for j in range(len(p_atual)):
				m.append((p_ant[j] - p_atual[j])/(0 - t))

			while(timer < t):
				timer += self.deltaTime
				for j in range(len(p_atual)):
					self.angulos[j] = m[j]*timer + p_ant[j]
		self.state = 'IDLE'
		self.interpolando = False


	def atualiza_cinematica(self):
		x = (self.t_state * self.nEstados)/self.tempoPasso
		pelv_point, foot_point = self.getTragectoryPoint(x)
		data_pelv = self.footToHip(pelv_point)
		data_foot = self.footToHip(foot_point)
		tanh_arg = (x-self.nEstados/2)/25
		# aux = self.angulo_vira/2 * tgh(tanh_arg)
		# aux = self.angulo_vira/2.*((np.exp(tanh_arg) - np.exp(-tanh_arg))/(np.exp(tanh_arg)+np.exp(-tanh_arg)))

		aux = self.angulo_vira/2. * math.tanh(tanh_arg)

		angulo_vira_plus_aux = self.angulo_vira/2. + aux # [0, self.angulo_vira]
		angulo_vira_minus_aux = self.angulo_vira/2. - aux # [-self.angulo_vira, 0]
		# angulo positivo: rotacao p esquerda
		# rota_dir: rotacao perna direita
		# rota_esq: rotacao perna esquerda
		# se perna direita no chao (self.perna = 1):
			# rota_dir%2 == self.perna -> gira perna direita
			# rota_esq%2 != self.perna -> gira perna esquerda
		# se perna esquerda no chao (self.perna = 0):
			# rota_dir%2 == self.perna -> gira perna direita
			# rota_dir%2 != self.perna -> girar perna esquerda
		if self.perna:
			data = data_pelv + data_foot + [0]*6
		else:
			data = data_foot + data_pelv + [0]*6
		# perna direita
		if (self.rota_dir != 0) and (self.rota_dir % 2 == self.perna):
			data[self.RIGHT_HIP_YALL] = copysign(angulo_vira_plus_aux, self.rota_dir) * DEG_TO_RAD
		if (self.rota_esq != 0) and (self.rota_esq % 2 != self.perna):
			data[self.LEFT_HIP_YALL] = copysign(angulo_vira_minus_aux, self.rota_esq) * DEG_TO_RAD
		#CONFIGURA BODY SOLVER PARA INVOCAR FUNÇÕES DO MODELO DINÂMICO DO ROBÔ
		ref_index = self.perna * 6 # 0 -> perna direita, primeiro; 6 -> perna esquerda primeiro
		self.body.set_angles(self.perna, data[0+ref_index:6+ref_index], data[6-ref_index:12-ref_index])

		data[0] = -data[0]
		data[4] = -data[4]

		offset = self.torsoOffsetMin + (self.torsoOffsetMax - self.torsoOffsetMin) * (self.deslocamentoYpelves/self.deslocamentoYpelvesMAX)
		data[self.RIGHT_HIP_PITCH] += offset
		data[self.LEFT_HIP_PITCH] += offset
		self.angulos = data


if __name__ == '__main__':
	control = Control(gravity_compensation_enable=True)
	control.run()
