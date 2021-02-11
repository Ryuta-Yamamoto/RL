from typing import Tuple, NamedTuple, List
import time
from dataclasses import dataclass
from itertools import chain
import random

import numpy as np
import gym
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

N_ACTIONS = 6

class State(NamedTuple):
    red_y: int
    green_y: int
    ball_x: int
    ball_y: int
    velocity_x: int
    velocity_y: int
    action: int

    @classmethod
    def len(cls) -> int:
        return len(cls._fields) + N_ACTIONS - 1

    def to_feature(self):
        lis = [0] * N_ACTIONS
        lis[self.action] = 1
        return list(self)[:-1] + lis


class Network(pl.LightningModule):
    def __init__(self, n_layers: int, n_nodes: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(State.len(), n_nodes),
            nn.ReLU()
        )
        self.layer = nn.Sequential(*chain(*(
            (nn.Linear(n_nodes, n_nodes), nn.ReLU()) for _ in range(n_layers - 1)
        )))
        self.output_layer = nn.Linear(n_nodes, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.layer(x)
        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.MSELoss()(y, y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


@dataclass(frozen=True)
class SnapShot:
    img: np.array
    pre_loc: Tuple[int, int] = (80, 80)

    @property
    def snipped_img(self) -> np.array:
        return self.img[34:-16].sum(axis=2)

    def red_loc(self) -> int:
        return (self.snipped_img[:, 16] == 417).argmax()

    def green_loc(self) -> int:
        return (self.snipped_img[:, 140] == 370).argmax()

    def ball_loc(self) -> Tuple[int, int]:
        arr = self.snipped_img == 708
        return arr.max(axis=1).argmax(), arr.max(axis=0).argmax()

    def ball_velocity(self) -> Tuple[int, int]:
        return (
            self.ball_loc()[0] - self.pre_loc[0],
            self.ball_loc()[1] - self.pre_loc[1]
        )

    def add_reward(self) -> float:
        y, x = self.ball_loc()
        return 1 - abs(y - self.green_loc()) * abs(x - 140) / 40000

    def state(self, action: int) -> State:
        return State(
            red_y=self.red_loc(),
            green_y=self.green_loc(),
            ball_x=self.ball_loc()[1],
            ball_y=self.ball_loc()[0],
            velocity_x=self.ball_velocity()[1],
            velocity_y=self.ball_velocity()[0],
            action=action,
        )

    def update(self, img: np.array) -> "SnapShot":
        return self.__class__(img=img, pre_loc=self.ball_loc())


@dataclass
class Agent:
    model: pl.LightningModule
    snapshot: SnapShot
    history: List[Tuple[State, float]]
    selected_action: int = 0
    gamma: float = 0.9
    epsilon: float = 1.0

    def observe(self, img: np.array, reward: float):
        self.snapshot = self.snapshot.update(img)
        self.history.append((self.snapshot.state(self.selected_action), reward + self.snapshot.add_reward()))

    def select(self) -> int:
        is_random = random.random() < self.epsilon
        action = random.randint(0, N_ACTIONS - 1) if is_random else self.best_action()
        self.selected_action = action
        return action

    def best_action(self) -> np.array:
        return np.array([
           model(torch.Tensor([self.snapshot.state(n).to_feature()])).detach().numpy()
           for n in range(N_ACTIONS)
        ]).argmax()

    def make_x(self) -> torch.Tensor:
        return torch.Tensor([state.to_feature() for state, _ in self.history])

    def _q_target(self) -> torch.Tensor:
        X = self.make_x()
        r = torch.Tensor([[reward] for _, reward in self.history])
        return r + X * self.gamma

    def make_dataset(self) -> TensorDataset:
        return TensorDataset(self.make_x(), self._q_target())

    def update(self):
        dataset = self.make_dataset()
        val_length = int(len(dataset) * 0.1)
        train, val = random_split(dataset, [len(dataset) - val_length, val_length])
        trainer = pl.Trainer(max_epochs=3)
        trainer.fit(self.model, DataLoader(train), DataLoader(val))
        self.epsilon *= 0.9


if __name__ == "__main__":
    env = gym.make('Pong-v0')
    img = env.reset()
    model = Network(3, 10)
    agent = Agent(model, snapshot=SnapShot(img), history=[])
    for n in range(10):
        env.reset()
        done = False
        while not done:
        # for n in range(100):
            # 1st dim [34:-16]
            # 相手の色(R+G+B) -> 417 dim2=16
            # 自分の色(R+G+B) -> 370 dim2=140
            # 玉の色(R+G+B)   -> 708
            env.render()
            action = agent.select()
            # action = agent.best_action()
            img, reward, done, info = env.step(action)
            agent.observe(img, reward)
            # print(info)
            # print(obs)
            # print(reward)
        agent.update()
