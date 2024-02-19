import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as D
import flax
from matplotlib import pyplot as plt
from matplotlib import patches
from gym import spaces


class InvertedPendulum:
    def __init__(self):
        init = np.array([0.3, 0.3], np.float32)
        self.init_spaces = [spaces.Box(low=-init, high=init, dtype=np.float32)]
        init = np.array([-1, 1], np.float32)
        self.init_spaces_train = [
            spaces.Box(low=-init, high=init, dtype=np.float32)]

        high = np.array([3, 3], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)
        self.noise = np.array([0.02, 0.01])

        safe = np.array([0.2, 0.2], np.float32)
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)
        safe = np.array([0.1, 0.1], np.float32)
        self.safe_space_train = spaces.Box(
            low=-safe, high=safe, dtype=np.float32)

        # reach_space = np.array([1.5, 1.5], np.float32)  # make it fail
        reach_space = np.array([0.7, 0.7], np.float32)
        # reach_space = np.array([0.5, 0.5], np.float32)  # same as in AAAI
        self.reach_space = spaces.Box(
            low=-reach_space, high=reach_space, dtype=np.float32
        )

        self.unsafe_spaces = [
            spaces.Box(
                # [-0.7, -0.7]
                low=self.reach_space.low,
                # [-0.6, 0]
                high=np.array([self.reach_space.low[0] + 0.1, 0.0]),
                dtype=np.float32,
            ),
            spaces.Box(
                # [0.6, 0]
                low=np.array([self.reach_space.high[0] - 0.1, 0.0]),
                # [0.7, 0.7]
                high=self.reach_space.high,
                dtype=np.float32,
            ),
        ]

    def next(self, state, action):
        th, thdot = state[:, 0], state[:, 1]  # th := theta
        max_speed = 5
        dt = 0.05
        g = 10
        m = 0.15
        l_ = 0.5
        b = 0.1
        u = 2 * torch.clip(action, -1, 1)[:, 0]
        newthdot = (1 - b) * thdot + (
            -3 * g * 0.5 / (2 * l_) * torch.sin(th + torch.pi)
            + 3.0 / (m * l_ ** 2) * u
        ) * dt
        newthdot = torch.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt
        newth = torch.clip(
            newth, self.reach_space.low[0], self.reach_space.high[0])
        newthdot = torch.clip(
            newthdot, self.reach_space.low[1], self.reach_space.high[1])
        newth = torch.unsqueeze(newth, dim=1)
        newthdot = torch.unsqueeze(newthdot, dim=1)
        return torch.hstack([newth, newthdot])


def in_(b, p):
    res = torch.ones(p.shape[0])
    for i in range(p.shape[-1]):
        res = torch.logical_and(res, p[:, i] >= b.low[i])
        res = torch.logical_and(res, p[:, i] <= b.high[i])
    return res


def jax_load(params, filename):
    with open(filename, "rb") as f:
        bytes_v = f.read()
    params = flax.serialization.from_bytes(params, bytes_v)
    return params


p = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

params = jax_load(None, 'checkpoints/pend_100_ppo.jax')
params = params['policy']['params']['params']

with torch.no_grad():
    p[0].weight = nn.Parameter(
        torch.tensor(params['Dense_0']['kernel']).T)
    p[2].weight = nn.Parameter(
        torch.tensor(params['Dense_1']['kernel']).T)
    p[4].weight = nn.Parameter(
        torch.tensor(params['Dense_2']['kernel']).T)
    p[0].bias = nn.Parameter(
        torch.tensor(params['Dense_0']['bias']))
    p[2].bias = nn.Parameter(
        torch.tensor(params['Dense_1']['bias']))
    p[4].bias = nn.Parameter(
        torch.tensor(params['Dense_2']['bias']))

pend = InvertedPendulum()

print(pend.unsafe_spaces)


x = torch.rand(3000, 2)
x_min, x_max = pend.reach_space.low, pend.reach_space.high
for i in range(2):
    x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

mask = ~in_(pend.safe_space, x)
mask.logical_and_(~in_(pend.unsafe_spaces[0], x))
mask.logical_and_(~in_(pend.unsafe_spaces[1], x))
x = x[mask]


# SIMULATION SECTION
X = []
X.append(x)

plt.ion()
fig, ax = plt.subplots()


def box2rect(b: spaces.Box, edgecolor, facecolor):
    diff = b.high - b.low
    return patches.Rectangle(
        b.low, diff[0], diff[1],
        linewidth=1, edgecolor=edgecolor, facecolor=facecolor
    )


sc = ax.scatter(x[:, 0], x[:, 1], s=1)
unsafe1 = box2rect(pend.unsafe_spaces[0], 'red', 'white')
unsafe2 = box2rect(pend.unsafe_spaces[1], 'red', 'white')
safe = box2rect(pend.safe_space, 'green', 'white')
ax.add_patch(unsafe1)
ax.add_patch(unsafe2)
ax.add_patch(safe)


plt.pause(0.1)
for i in range(50):
    print(f'=== step {i} ===')
    x = X[-1]
    x_nxt = pend.next(x, p(x)).detach()
    X.append(x_nxt)
    sc.set_sizes(None)
    sc = ax.scatter(x[:, 0], x[:, 1], s=1)
    reach = in_(pend.safe_space_train, x).sum()
    unsafe = (
        in_(pend.unsafe_spaces[0], x)
        + in_(pend.unsafe_spaces[1], x)
    ).sum()
    print(f'reach = {reach}')
    if unsafe:
        print(f'unsafe = {unsafe}')
    plt.draw()
    plt.pause(0.1)

plt.close()
plt.ioff()


class Learner_ReachAvoid:
    def __init__(self, n_dims, models):
        # Abstraction A comes with a delta; so A is a non-deterministic
        # transition system.
        self.n_dims = n_dims
        self.V = models[0]

    def init_optimizer(self, lr):
        return torch.optim.SGD(
            list(self.V.parameters()), lr=lr)

    def loss(self, x):
        """Aggregate loss function for the certificate NN.

        New components can be added to the loss by defining new
        functions and calling them in the expression evaluated below.
        """
        # Adding delta * ball to account for non-determinism
        x_nxt = pend.next(x, p(x))
        l_dec = torch.relu(self.V(x_nxt) - self.V(x) + 1).mean()
        l_safe1 = torch.relu(self.V(x) - 1e2).mean()
        l_safe2 = torch.relu(self.V(x_nxt) - 1e2).mean()
        nxt_unsafe = (
            in_(pend.unsafe_spaces[0], x_nxt)
            + in_(pend.unsafe_spaces[1], x_nxt))
        l_nxt_unsafe = (
            torch.relu(1e2 - self.V(x_nxt)) * nxt_unsafe).mean()
        return l_dec + l_safe1 + l_safe2 + l_nxt_unsafe

    def chk(self, x):
        x_nxt = pend.next(x, p(x))
        c_dec = self.V(x) >= self.V(x_nxt) + 1e-3
        c_safe1 = self.V(x) <= 1e2
        c_safe2 = self.V(x_nxt) <= 1e2
        nxt_unsafe = (
            in_(pend.unsafe_spaces[0], x_nxt)
            + in_(pend.unsafe_spaces[1], x_nxt))
        c_nxt_unsafe = (
            (self.V(x_nxt) >= 1e2) * nxt_unsafe
            + 1 * ~nxt_unsafe)
        return (c_dec * c_safe1 * c_safe2 * c_nxt_unsafe).float().mean() * 100

    def fit(self, S, n_epoch=512, batch_size=100, lr=1e-3, gamma=1.0):
        def training_step(s, optimizer):
            optimizer.zero_grad()
            loss = self.loss(s)
            chk = self.chk(s)
            loss.backward(retain_graph=True)
            optimizer.step()
            return loss, chk

        def training_loop():
            optimizer = self.init_optimizer(lr)
            loader = D.DataLoader(S, batch_size=batch_size, shuffle=True)
            for e in range(n_epoch + 1):
                for b_idx, s in enumerate(loader):
                    loss, chk = training_step(s, optimizer)
                print(
                    f'Epoch {e:>6}, '
                    + f'Loss={self.loss(S):12.6f}, '
                    + f'Chk={self.chk(S):12.6f}, ')
                if self.chk(S) == 100.0:
                    return

        training_loop()


def nn_ReachAvoid(n_dims):
    return nn.Sequential(
        nn.Linear(n_dims, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Softplus()
    )


x = X[0]
learner = Learner_ReachAvoid(2, [nn_ReachAvoid(2)])
learner.fit(x, n_epoch=1024)
