---
layout: post
title: gymnasium 패키지로부터 독립하기(2)
date: 2024-06-08 10:00 +0900
tags: [pipenv, 강화학습, vscode, tutorial, PyTorch, Gymnasium, CartPole, debug, Jupyter Notebook]
categories: 강화학습
toc: true
---

## 2nd reset

두 번째 reset 함수가 episode for문 안에 들어있습니다.
env.reset() 함수를 우리의 함수로 바꿔주겠습니다.
```python
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = my_reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
```
<br/>

## Original step & init
마지막으로 env 객체가 사용된 곳은 step 함수가 있는 곳입니다.
step 함수는 이름에서 알 수 있듯이 한 번의 action이 수행되면 다음 상태를 구하는 함수입니다.
즉, 다음 한 스텝 진행되는 것을 의미합니다.
하지만 이 step 함수를 이해하기 위해서는 먼저 초기화를 수행하는 init 함수도 알아야 합니다.

먼저 cartpole.py 내부를 살펴볼까요?
CartPoleEnv 클래스 내부에 init 함수와 step 함수가 보입니다.

```python
class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
...
    def __init__(self, render_mode: Optional[str] = None):
            self.gravity = 9.8
            self.masscart = 1.0
            self.masspole = 0.1
            self.total_mass = self.masspole + self.masscart
            self.length = 0.5  # actually half the pole's length
            self.polemass_length = self.masspole * self.length
            self.force_mag = 10.0
            self.tau = 0.02  # seconds between state updates
            self.kinematics_integrator = "euler"

            # Angle at which to fail the episode
            self.theta_threshold_radians = 12 * 2 * math.pi / 360
            self.x_threshold = 2.4

            # Angle limit set to 2 * theta_threshold_radians so failing observation
            # is still within bounds.
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            )

            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

            self.render_mode = render_mode

            self.screen_width = 600
            self.screen_height = 400
            self.screen = None
            self.clock = None
            self.isopen = True
            self.state = None

            self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
```

코드를 천천히 살펴보면 어떠신가요?
생각보다는 간단하게 구현되어있지 않나요?
이 정도는 아주 쉽게 우리의 코드로 가져올 수 있을 것 같습니다.
<br>

## Init

먼저 init 함수부터 가져와보겠습니다.
나만의 함수를 만들어서 한번 수행하는 것도 좋은 방법처럼 보입니다.
하지만 함수 자체가 너무 간단하여 그렇게 하기 귀찮다면 그냥 바로 변수들을 초기화 하는 것도 나쁘지 않아 보입니다.
그래서 저는 cuda 사용 확인 if문과 episode for문 사이에 아래처럼 변수 초기화를 해보았습니다.
```python
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = "euler"

theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4
steps_beyond_terminated = None

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    ...
```

env라는 환경 객체를 여러개 생성할 필요는 없어보입니다.
우리는 하나의 cartpole 게임만 관심이 있기 때문입니다.
따라서 "self." 구문은 전부 삭제해도 되겠군요.

과연 spaces 모듈에서 파생된 객체들이 필요할까요?
필요없다는 것을 눈치 채셨다면 대단한 코딩 감각을 가지셨다고 말씀드릴 수 있겠네요.
Discrete 객체와 Box 객체는 실질적으로 강화학습을 수행하는데 아무 필요가 없습니다.
이 두 객체는 이 cartpole이라는 환경에대해 설명하는 일종의 metadata의 역할을 수행합니다.
따라서 high라는 array도 상태 변수가 가질 수 있는 최대, 최솟값을 나타내기 위한 수단일 뿐입니다.
과감하게 모두 삭제해 주겠습니다.

다음으로 랜더링 관련 변수들입니다.
랜더링은 사용하지 않는 것으로 가정했으므로 모두 삭제해 주겠습니다.

init 함수 가장 마지막 줄에 또 등장한 steps_beyond_terminated 변수가 있습니다.
다시 한번 설명은 뒤로 미루도록 하겠습니다.
<br>

## Step

드디어 마지막 step() 함수입니다.