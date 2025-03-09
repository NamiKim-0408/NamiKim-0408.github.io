---
layout: post
title: 나만의 gymnasium 패키지 만들기(1)
date: 2025-03-09 10:00 +0900
tags: [pipenv, 강화학습, vscode, tutorial, PyTorch, Gymnasium, CartPole, debug, Jupyter Notebook]
categories: 강화학습
toc: true
---

## 1
이번 포스트에서는 나만의 커스텀 환경을 만드는 방법을 살펴볼 예정입니다.
이 포스트의 내용은 거의 [Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/) 이 페이지의 내용을 번역, 해설한 내용에 제가 필요하다고 생각하는 내용을 덧 붙인 글이 될 것입니다.
따라서 원문을 보고 자세히 공부해 보는 것도 좋을 듯합니다.

이 Gymnasium 문서에서 고려하는, custom 환경으로 만들고자 하는 게임은 <mark>GridWorldEnv</mark>라는 게임이군요.
이 게임이 무엇인지 먼저 살펴보겠습니다.

<br/>
> Basic information about the game
> * Observations provide the location of the target and agent.
> * There are 4 discrete actions in our environment, corresponding to the movements “right”, “up”, “left”, and “down”.
> * The environment ends (terminates) when the agent has navigated to the grid cell where the target is located.
> * The agent is only rewarded when it reaches the target, i.e., the reward is one when the agent reaches the target and zero otherwise.
<br/>

<p align="center">
  <img src="/assets/img/wgym_1/example.gif">
</p>
<center>Example</center>

위 이미지를 보시면 딱 어떤 게임인지 감이 오실거라 믿습니다.
파란원으로 표현된 agent가 빨간 네모로 표현된 target을 찾아가는 게임이군요.
주목할 만한 점은 이 게임은 cartpole과는 다르게 reword가 최종 target에 도달 했을 때만 주어진다는 점입니다.
이처럼 지연된 보상을 주는 강화학습 게임들이 많이 존재하고 이 게임도 그 중 하나입니다.


## __init__
이전 gymnasium 패키지로부터 독립하기 포스트를 읽어보신 분은 cartpole 클래스 내부가 어떻게 구성되어 있는지 기억하실 겁니다.
클래스 객체가 생생될 때 실행되어야 할 __init__ 함수, step 함수 등이 구현되어 있었습니다.
마찬가지로 custom 게임을 구현할 때 이런 함수들을 직접 구현하여야 합니다.
이 절에서는 __init__ 함수를 살펴 보겠습니다.

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
```

이 GridWorld 환경을 만들기 위해서는 gym의 기본 추상 클래스인 <mark>gymnasium.Env</mark>를 상속받아야 한다고 메뉴얼에서 설명하고 있습니다.
```python
class GridWorldEnv(gym.Env):
```