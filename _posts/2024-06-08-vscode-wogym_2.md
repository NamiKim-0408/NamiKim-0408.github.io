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

클래스를 이용하여 env라는 환경 객체를 여러개 생성할 필요는 없어보입니다.
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
episode for 문이 시작하기 전에 적당한 위치에 my_step() 함수의 선언문을 작성해주겠습니다.
입력 매개변수는 일단 self는 삭제하고 action만 두겠습니다.
그리고 cartpole env의 setp() 함수를 복사합니다.

gymnasium 패키지의 cartpole의 step() 함수의 시작은 assert를 이용하여 valid check를 먼저 수행하는 군요.
우리는 action_space와 state를 잘 정의하여 쓸 예정이니 이 부분 역시 과감히 삭제하겠습니다.
삭제되는 코드는 아래 4줄과 같습니다.
```python
assert self.action_space.contains(
    action
), f"{action!r} ({type(action)}) invalid"
assert self.state is not None, "Call reset before using step method."
```

마찬가지로 "self." 키워드는 모두 삭제하겠습니다.
어차피 클래스를 사용하지 않아 삭제하지 않으면 오류를 내뱉습니다.

또 함수 안에 라이브러리 log 함수인 logger.warn() 함수가 있습니다.
간단하게 print() 함수로 바꿔주도록 하겠습니다.

다음으로 return 직전에 render 관련 코드가 2줄 있습니다.
```python
if self.render_mode == "human":
    self.render()
```
이 두줄은 삭제하겠습니다.

일단 1차적인 정리는 끝난듯 합니다.
다음으로 고쳐하 할 부분은 state 변수입니다.
state 변수는 gymnasium 패키지를 그대로 사용했다면 패키지에서 알아서 관리해 주었겠지만 우리의 수정 코드에서는 직접 관리하고 함수 내부로 넘겨주어야 합니다.
문제가되는 부분은 
```python
x, x_dot, theta, theta_dot = state
```
이 부분이겠네요.
그럼 이 state 변수를 함수 내부로 전달하기 위해 매개변수를 설정하겠습니다.
함수 선언을 다음과 같이 고치겠습니다.
```python
def my_step(action, state):
```
이제 my_step 함수는 현재 상태를 입력으로 받아 다음 스텝을 진행할 수 있게되었습니다.
함수를 부르는 부분도 state를 넣어서 불러보겠습니다.
```python
for t in count():
    action = select_action(state)
    observation, reward, terminated, truncated, _ = my_step(action.item(), state)
    reward = torch.tensor([reward], device=device)
```
그럼 이제 실행하면 될까요?
안타깝게도 코드는 정상동작 하지 않는데 입력 변수 형(type)이 맞지 않아서 입니다.

pytorch를 사용할 때 형변환은 초보자가 넘어야 할 큰 산 중 하나입니다.
cpu <-> GPU 메모리 공간, tensor <-> numpy 변환, squeeze/unsqueeze 함수를 통한 차원 축소/확장 이 개념들은 반드시 알아야 하는 개념으로 충분히 숙지하시길 바랍니다.

그럼 먼저 위코드에서 형 변환이 이루어진 예시 코드를 먼저 볼까요?
```python
for t in count():
    action = select_action(state)
    observation, reward, terminated, truncated, _ = my_step(action.item(), state.cpu().numpy().squeeze())
    reward = torch.tensor([reward], device=device)
```
간단히 하나씩 설명하자면, cpu()함수는 GPU에 저장된 tensor를 cpu로 (정확하게는 컴퓨터의 RAM이겠죠?) 불러오는 함수입니다.
numpy는 tensor를 numpy array로 변환하는 함수입니다.
사실 이번 예시에서는 numpy() 함수는 굳이 사용하지 않아도 동작하지만 예시를 위해 넣어보았습니다.
squeeze() 함수는 차원을 하나 축소하는 함수입니다.
squeeze 하기전 state 변수는 [1,4]의 크기를 가지는 매트릭스입니다.
squeeze로 차원 축소를 하면 크기 4의 벡터로 변환되고 비로소 x, x_dot, theta, theta_dot 네개의 변수로 unzip이 가능해집니다.
<br/>

## steps_beyond_terminated
이제 정말 마지막 단계만 앞두고있습니다.
계속 해서 설명을 미뤄왔던 steps_beyond_terminated 변수를 한번 알아보겠습니다.
이 변수의 정체는 원본 step() 함수 내부를 살펴보면 알 수 있습니다.

step() 함수 내부에서 게임의 종료 조건, 즉 막대기가 이번 스텝에서 쓰러지면 terminated 변수에 true 값을 저장하여 표시합니다.
steps_beyond_terminated 변수는 막대기가 쓰러지면 최초 0의 값을 할당받고 이렇게 쓰러진 상태에서 계속 step 함수가 불려지면 += 1 동작을 수행하는 군요.
아무래도 쓰러지고 난 이후 계속되는 게임이 끝나지 않는 상황 등 특수한 상황에서 사용하기 위해 만들어진 변수로 추측됩니다.
우리의 예제에서는 이러한 경우를 고려하지 않으므로 사실 steps_beyond_terminated 관련 부분을 모두 삭제하여도 무방합니다.

이번 예제에서는 삭제하지 말고 간단히 수정하여 코드가 정상 동작하도록 해보겠습니다.
먼저 my_reset() 함수를 살작 수정해보겠습니다.
reset이 진행될 때, steps_beyond_terminated 변수도 reset 해주겠습니다.
```python
def my_reset():
    state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    return np.array(state, dtype=np.float32), {}, None
state, info, steps_beyond_terminated = my_reset()
```

다음으로 my_step() 함수의 매개변수로 하나 추가해주고 또 반환값으로도 추가해 보겠습니다.
```python
def my_step(action, state, steps_beyond_terminated):
...
    return np.array(state, dtype=np.float32), reward, terminated, False, steps_beyond_terminated
```
```python
observation, reward, terminated, truncated, steps_beyond_terminated = my_step(action.item(), state.cpu().squeeze(), steps_beyond_terminated)
```
이제 steps_beyond_terminated 변수는 에피소드가 시작할 때 마다 None 값으로 reset되고 막대가 쓰러지면 0 값을 가지는 변수가 되었습니다.
<br/>

## 마치며
여기까지 따라오시느라 수고하셨습니다.
이제 F5키를 눌러 학습을 시작해 볼까요?
우리의 강화학습 머신이 gymnasium 패키지 없이도 cartpole 게임을 잘 학습하는 모습을 보실 수 있습니다.
이제 우리는 이 코드를 베이스로 모든 문제를 강화학습을 통해 해결할 수 있게 되었습니다.
간단히 혹은 복잡하게 reset, step 함수를 수정하고 초기화만 수행해 주면 학습이 이루어 질 것입니다.

제가 작성한 최종 코드를 참고하시고 싶으시면 [다운로드][1]해주세요. 

[1]:http://namikim-0408.github.io/download/ai_withoutGym.py