import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np
import matplotlib.pyplot as plt
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations

class first_project(Node):

    def __init__(self):
        super().__init__('first_project')
        self.get_logger().debug('Definido o nome do nó para "R2D2"')

        # QoS settings for reliability
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        # Publisher para o controle de velocidade do robô
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.on_timer)

        self.laser = None
        self.pose = None

        self.ir_para_frente = Twist(linear=Vector3(x=0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))
        self.parar = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))

        # Variaveis do filtro de Kalman
        self.position_estimate_kalman = 10.0  # Assumindo que o robô plota no meio (x = 10)
        self.position_covariance_kalman = 1.0 # Valor inicial escolhido

        # Variaveis para localizacao de Markov
        self.position_estimate_markov = 10.0  # Assumindo que o robô plota no meio (x = 10)
        self.transition_prob = 0.9  # Probabilidade de se manter no mesmo estado
        self.measurement_prob = 0.7  # Probabilidade da medição estar correta

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges

    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose

    def on_timer(self):
       
        self.pub_cmd_vel.publish(self.ir_para_frente)
    def run(self):
        self.get_logger().info('Iniciando movimento: ir para frente')
        self.pub_cmd_vel.publish(self.ir_para_frente)

        while rclpy.ok():
            rclpy.spin_once(self)

            # Get sensor data
            if self.laser is None or self.pose is None:
                continue

            # Update distances
            self.distancia_frente = min(self.laser[80:100])
            self.get_logger().info(f"Distância para o obstáculo: {self.distancia_frente}")

            # If obstacle detected, stop
            if self.distancia_frente < 1.5:
                self.get_logger().info('Obstáculo detectado. Parando o robô.')
                self.pub_cmd_vel.publish(self.parar)
                rclpy.spin_once(self)
                break

            # Update position estimates
            self.kalman_filter_update()
            self.markov_localization_update()

        self.get_logger().info('Finalizando movimento do robô.')

    def kalman_filter_update(self):
        # Obtém a posição medida pelo sensor de odometria
        measurement_position = self.pose.position.x

        # Etapa de previsão: A posição prevista é a estimativa anterior
        predicted_position = self.position_estimate_kalman
        
        # Adiciona o ruído do processo à covariância (incerteza) prevista
        predicted_covariance = self.position_covariance_kalman + 0.1  # Ruído do processo (Rt = 0.1)

        # Cálculo do ganho de Kalman: define o peso da medição
        kalman_gain = predicted_covariance / (predicted_covariance + 1.0)  # Ruído da medição (Qt = 1.0)

        # Atualiza a estimativa da posição com base no ganho de Kalman e no erro da medição
        self.position_estimate_kalman = predicted_position + kalman_gain * (measurement_position - predicted_position)

        # Atualiza a incerteza (covariância) da estimativa com base no ganho de Kalman
        self.position_covariance_kalman = (1 - kalman_gain) * predicted_covariance

        # Exibe a nova posição estimada pelo filtro de Kalman no log
        self.get_logger().info(f"Posição estimada (Kalman): {self.position_estimate_kalman}")

        self.plot_gaussian(self.position_estimate_kalman, self.position_covariance_kalman, method="Kalman Filter")

    def markov_localization_update(self):
        # Obtém a posição medida pelo sensor de odometria
        measurement_position = self.pose.position.x

        # Verifica se a medição está próxima o suficiente da estimativa anterior (diferença menor que 0.5 metros)
        if abs(measurement_position - self.position_estimate_markov) < 0.5:
            # Atualiza a estimativa de posição usando a probabilidade de transição (estado permanece o mesmo)
            self.position_estimate_markov = (self.transition_prob * self.position_estimate_markov +
                                            (1 - self.transition_prob) * measurement_position)
        else:
            # Se a medição estiver distante, atualiza a posição com base na probabilidade de medição
            self.position_estimate_markov = (1 - self.measurement_prob) * self.position_estimate_markov + \
                                            self.measurement_prob * measurement_position

        # Exibe a nova posição estimada pelo algoritmo de Markov no log
        self.get_logger().info(f"Posição estimada (Markov): {self.position_estimate_markov}")

        # Desenha a curva Gaussiana com a nova posição estimada (variância fixa de 1.0)
        #self.plot_gaussian(self.position_estimate_markov, 1.0, method="Markov Localization")

    def plot_gaussian(self, mean, variance, method="Method"):
        # Eixo x de 0 a 10 (O robô já incia no meio)
        x = np.linspace(0, 10, 1000)
        gaussian = (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

        # Plot Gaussian
        plt.figure()
        plt.plot(x, gaussian)
        plt.title(f"Posição Estimada: {mean:.2f}, Variância: {variance:.2f} ({method})")
        plt.xlabel('Posição (metros)')
        plt.ylabel('Probabilidade')
        plt.grid()
        plt.show()

# Main function
def main(args=None):
    rclpy.init(args=args)
    node = first_project()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass        

if __name__ == '__main__':
    main()
