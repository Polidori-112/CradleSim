from Pendulum import*


class Simulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ball_radius = width * 0.025
        self.pendulums = []
        right_strings = [0.6]
        left_strings = [0.4, 0.45, 0.5, 0.55]
        strings = [0.4, 0.45, 0.5, 0.55, 0.6]
        colors = ['white', 'white', 'white', 'white', 'white']
        self.pendulum_init(colors, left_strings, 0)
        self.pendulum_init(['white'], right_strings, math.pi / 2)

    def pendulum_init(self, colors, strings, starting_theta):
        for index in range(0, len(strings)):
            self.pendulums.append\
            (Pendulum(self.width * strings[index], self.height * 0.5,
                      radius=self.ball_radius, color=colors[index],
                      a=starting_theta))


    def moving_logic(self, surface):
        for pendulum in self.pendulums:
            pendulum.step()
            pendulum.draw(surface)

    def distance(self, pendulum1, pendulum2):
        x1 = pendulum1.x
        x2 = pendulum2.x
        y1 = pendulum1.y
        y2 = pendulum2.y
        x_dif = x2 - x1
        y_dif = y2 - y1
        dist = np.sqrt(x_dif**2 + y_dif**2)
        return dist

    def collision_logic(self):
        for index in range(0, len(self.pendulums) - 1):
            dist = self.distance(self.pendulums[index], self.pendulums[index + 1])
            total_av = self.pendulums[index].av + self.pendulums[index + 1].av
            if dist < (self.pendulums[index].radius * 2):
                # Swap Angular Velocity
                self.pendulums[index].av, self.pendulums[index + 1].av =\
                    self.pendulums[index + 1].av, self.pendulums[index].av
