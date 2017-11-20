from .negotiablegame import NegotiableGame
from nrl.envs.marsexplorer import *
import itertools
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cs

cdict = {'red':   ((0.0,  0.173, 0.173),
                   (1.0,  0.925, 0.925)),

         'green': ((0.0,  0.067, 0.067),
                   (1.0, 0.384, 0.384)),

         'blue':  ((0.0,  0.027, 0.027),
                   (1.0,  0.196, 0.196))}
plt.register_cmap(name='RustPlanet', data=cdict)
REWARD_COLORS = cm.get_cmap('RustPlanet')
MAP_COLORS = {b'B':"#3a0e00",
              b'F':"#933111",
              b'S':"#933111",
              b'U':"#d65b33",}


class NegotiableMarsExplorer(NegotiableGame):


    def __init__(self, mdps, true_mdp=None, ws=None, gamma=0.95, matrix_form=True, render_scale=1):
        self.scale = render_scale
        self.tt_rewards = np.zeros(self.I)
        super(NegotiableMarsExplorer, self).__init__(mdps,true_mdp, ws, gamma, matrix_form)


    def act(self, action):
        """
        Implements an action in the game. Updates the world_state, belief, and
        true/individual rewards.
        :param action: the robot's action.
        """
        s = self.world_state
        sprime, r, done, pdict = self.true_mdp.step(action)
        self.world_state = sprime
        self.updateBelief(s, action, sprime)
        self.true_reward += r
        self.discounted_reward += r * self.gamma ** self.time_step
        self.separate_rewards += np.array([self.getReward((s,theta),action,(sprime, theta)) for theta in self.getAllTheta()])
        self.tt_rewards += np.array([((s,sprime) in mdp.true_goal_transitions) for mdp in self.mdps]) * self.gamma ** self.time_step
        self.time_step += 1
        return sprime, done, pdict['prob']

    def reset(self):
        self.tt_rewards = np.zeros(self.I)
        super(NegotiableMarsExplorer, self).reset()


############ Visualizations #############

    def _close_view(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True

    def render(self, mode='human', close=False):
        if close:
            self._close_view()
            return

        width = self.true_mdp.ncol
        height = self.true_mdp.nrow
        canvas_width = width * self.scale
        canvas_height = height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        self.canvas.delete(tk.ALL)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill='black')

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + 1) * self.scale,
                (y + 1) * self.scale,
                fill=color,
            )

        for x in range(width):
            for y in range(height):
                if self.beams[x, y] == 1:
                    fill_cell(x, y, 'yellow')
                if self.food[x, y] == 1:
                    fill_cell(x, y, 'green')
                if self.walls[x, y] == 1:
                    fill_cell(x, y, 'grey')

        for i, (x, y) in enumerate(self.agents):
            if not self.tagged[i]:
                fill_cell(x, y, self.agent_colors[i])

        self.root.update()

    def _close(self):
        self._close_view()
