---
layout: post
title: 나만의 gymnasium 패키지 만들기(2)
date: 2025-03-16 10:00 +0900
tags: [pipenv, 강화학습, vscode, tutorial, PyTorch, Gymnasium, CartPole, debug, Jupyter Notebook]
categories: 강화학습
toc: true
---

## Intro
이전 포스트까지 우리는 custom 환경을 만드는 법을 알아봤습니다.
이제는 만들어진 환경을 이용해서 실제 강화학습 게임을 진행해 보겠습니다.
사실 거의 대부분의 코드를 cartpole에서 가져올 예정입니다.
즉, cartpole 예제에서 환경만 gridworld로 바꾸는 것이겠죠.
하지만 이렇게 하기 위해 조금씩 손봐야 할 부분이 있습니다.

## Wrappers
전 포스트에서 observation 공간을 어떤 클래스로 구현했는지 기억하시나요?
공식 예제에서는 __Dict__ 클래스를 이용하여 구현하였고 이 클래스는 Dictionary 자료형입니다.
이전 포스트에서 observation 공간을 정의하는 부분을 다시 보겠습니다.
```python
self.observation_space = gym.spaces.Dict(
    {
        "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
    }
)
```
__observation_space__ 라는 변수에는 dictionary 인스턴스를 넣었습니다.
이 dictionary 인스턴스는 내부에 "agent", "target"이라는 키를 두개 가지고 각 키에 대응하는 값은 <mark>Box</mark> 클래스로 구현하였습니다.
이 박스 클래스 역시 <mark>spaces</mark> 모듈의 하위 클래스인 것을 볼 수 있습니다.

[Wrappers](https://gymnasium.farama.org/api/wrappers/) 모듈은 무엇일까요?
공식 메뉴얼에 따르면 이 모듈은 Gymnasium의 공식 환경 혹은 다른 비공식 환경을 수정할 수 있는 모듈입니다.
이 Wrappers 모듈을 이용하면 다른 특별한 수정 없이 간단히 필요한 부분을 수정할 수 있다고 합니다.
특히나 <mark>make()</mark> 함수를 이용해 이미 만들어진 환경도 바로 수정할 수 있습니다.
공식 메뉴얼에서 Gridworld의 observation 공간을 처음부터 <mark>Box</mark>로 구현하지 않고 <mark>Dict</mark>으로 구현한 것은 아마 몰라서가 아니라 이 Wrappers 모듈을 설명하기위해서가 아니었을까 추측해 봅니다.
```python
from gymnasium.wrappers import FlattenObservation

env = gym.make('gymnasium_env/GridWorld-v0')
env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))
env.reset()
({'agent': array([4, 1]), 'target': array([2, 4])}, {'distance': 5.0})
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space
Box(0, 4, (4,), int64)
wrapped_env.reset()
(array([3, 0, 2, 1]), {'distance': 2.0})
```

우리의 환경을 학습 코드에 사용할때 observation 공간은 Dict 자료형 보단 Box 형태로 사용되어한다고 합니다.
이 Dict 자료형의 각각의 벡터를 이어주는 함수가 __FlattenObservation__ 함수이고 바로 적용하여 사용가능합니다.
그림에서 보시면 길이가 2인 Box 2개가 길이가 4인 Box 하나로 펼쳐진 모습을 볼 수 있습니다.
Wrapper 모듈을 열심히 찾아보면 필요한 함수들을 찾을 수가 있겠군요.