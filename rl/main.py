import random
from typing import Tuple, NamedTuple, List, Optional, Union, Iterable
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

    def to_feature(self, action: Optional[int] = None) -> List[int]:
        lis = [0] * N_ACTIONS
        if action is None:
            action = self.action
        lis[action] = 1
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
    pre_vx: int = -1

    @property
    def snipped_img(self) -> np.array:
        return self.img[34:-16].sum(axis=2)

    def red_loc(self) -> int:
        return (self.snipped_img[:, 16] == 417).argmax()

    def green_loc(self) -> int:
        top = (self.snipped_img[:, 140] == 370).argmax()
        bottom = (self.snipped_img[:, 140] == 370).cumsum().argmax()
        return np.int64((top + bottom) / 2)

    def ball_loc(self) -> Tuple[int, int]:
        arr = self.snipped_img == 708
        return arr.max(axis=1).argmax(), arr.max(axis=0).argmax()

    def ball_velocity(self) -> Tuple[int, int]:
        return (
            self.ball_loc()[0] - self.pre_loc[0],
            self.ball_loc()[1] - self.pre_loc[1]
        )

    def is_end(self) -> bool:
        return sum(self.ball_velocity()) == 0

    def add_reward(self) -> float:
        y, x = self.ball_loc()
        if self.is_end():
            return -1
        base_score = 100 if (self.pre_vx > 0) & (self.vx < 0) else 1
        return base_score - abs(y - self.green_loc()) * abs(x - 140) / 40000

    @property
    def vx(self) -> int:
        return self.ball_velocity()[1]

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
        return self.__class__(img=img, pre_loc=self.ball_loc(), pre_vx=self.vx)


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

    def make_x(self, action: Union[None, int, Iterable[int]] = None) -> torch.Tensor:
        if action is None:
            action = [None] * len(self.history)
        if isinstance(action, int):
            action = [action] * len(self.history)
        return torch.Tensor([state.to_feature(a) for (state, _), a in zip(self.history, action)])

    def history_best_actions(self) -> torch.Tensor:
        X = torch.stack([self.make_x(a) for a in range(N_ACTIONS)])
        return self.model(X).argmax(axis=0)

    def _q_target(self) -> torch.Tensor:
        X = self.make_x(self.history_best_actions())
        r = torch.Tensor([[reward] for _, reward in self.history])
        with torch.no_grad():
            return r + self.model(X) * self.gamma

    def make_dataset(self) -> TensorDataset:
        return TensorDataset(self.make_x(), self._q_target())

    def update(self):
        dataset = self.make_dataset()
        val_length = int(len(dataset) * 0.1)
        train, val = random_split(dataset, [len(dataset) - val_length, val_length])
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, DataLoader(train, shuffle=True), DataLoader(val, shuffle=True))
        self.epsilon *= 0.95


if __name__ == "__main__":
    env = gym.make('Pong-v4')
    img = env.reset()
    model = Network(3, 10)
    agent = Agent(model, snapshot=SnapShot(img), history=[])
    histories = []
    for n in range(100):
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
            # print(
            # .info)
            # print(obs)
            # print(reward)
        histories += agent.history
        agent.history = list(set(agent.history) | set(random.choices(histories, k=1000)))
        agent.update()
        agent.history = []
