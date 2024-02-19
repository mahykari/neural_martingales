import numpy as np
import gym
import torch
import torch.nn as nn
import torch.utils.data as D
import flax
from matplotlib import pyplot as plt
from matplotlib import patches
from gym import spaces


def make_unsafe_spaces(obs_space, unsafe_bounds):
    unsafe_spaces = []
    dims = obs_space.shape[0]
    for i in range(dims):
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        high[i] = -unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(
                spaces.Box(low=low, high=high, dtype=np.float32))

        high = np.array(obs_space.high)
        low = np.array(obs_space.low)
        low[i] = unsafe_bounds[i]
        if not np.allclose(low, high):
            unsafe_spaces.append(
                spaces.Box(low=low, high=high, dtype=np.float32))
    return unsafe_spaces


def triangular(shape):
    U = torch.rand(*shape)
    p1 = -1 + torch.sqrt(2 * U)
    p2 = 1 - torch.sqrt((1 - U) * 2)
    return torch.where(U <= 0.5, p1, p2)


class CollisionAvoidanceEnv(gym.Env):
    name = "cavoid"

    def __init__(self):
        self.steps = None
        self.state = None
        self.has_render = False

        # init = np.array([1.0, 1.0], np.float32)
        # self.init_space = spaces.Box(low=-init, high=init, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        # was 0.05 before
        self.noise = np.array([0.05, 0.05], np.float32)  # was 0.02 before
        safe = np.array([0.2, 0.2], np.float32)  # was 0.1 before
        self.safe_space = spaces.Box(low=-safe, high=safe, dtype=np.float32)

        self.init_spaces_train = make_unsafe_spaces(
            self.observation_space, np.array([0.9, 0.9], np.float32)
        )
        self.init_spaces = [
            spaces.Box(
                low=np.array([-1, -0.6]),
                high=np.array([-0.9, 0.6]),
                dtype=np.float32,
            ),
            spaces.Box(
                low=np.array([0.9, -0.6]),
                high=np.array([1.0, 0.6]),
                dtype=np.float32,
            ),
        ]

        self.unsafe_spaces = []
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, 0.7]),
                high=np.array([0.3, 1.0]), dtype=np.float32
            )
        )
        self.unsafe_spaces.append(
            spaces.Box(
                low=np.array([-0.3, -1.0]),
                high=np.array([0.3, -0.7]), dtype=np.float32
            )
        )
        self.reach_space = self.observation_space
        # self.noise = np.array([0.001, 0.001])

    @property
    def noise_bounds(self):
        return -self.noise, self.noise

    def next(self, state, action):
        action = torch.clip(action, -1, 1)

        obstacle1 = torch.tensor((0, 1))
        force1 = torch.tensor((0, 1))
        dist1 = torch.norm(obstacle1 - state)
        dist1 = torch.clip(dist1 / 0.3, 0, 1)
        action = action * dist1 + (1 - dist1) * force1

        obstacle2 = torch.tensor((0, -1))
        force2 = torch.tensor((0, -1))
        dist2 = torch.norm(obstacle2 - state)
        dist2 = torch.clip(dist2 / 0.3, 0, 1)
        action = action * dist2 + (1 - dist2) * force2

        state = state + action * 0.2
        x_min, x_max = self.observation_space.low, self.observation_space.high
        x_min, x_max = torch.from_numpy(x_min), torch.from_numpy(x_max)
        state = torch.clip(
            state, x_min, x_max)

        return state

    def add_noise(self, state):
        noise = triangular(
            (state.shape[0], self.observation_space.shape[0],))
        noise1 = torch.from_numpy(self.noise)
        noise = noise * noise1
        return state + noise


def in_(b, p):
    res = torch.ones(p.shape[0])
    for i in range(p.shape[-1]):
        res = torch.logical_and(res, p[:, i] >= b.low[i])
        res = torch.logical_and(res, p[:, i] <= b.high[i])
    return res


def in2_(bs, p):
    res = torch.zeros(p.shape[0])
    for b in bs:
        res = torch.logical_or(res, in_(b, p))
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
    nn.Linear(128, 2),
)

params = jax_load(None, 'checkpoints/cavoid_100_ppo.jax')
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

cavoid = CollisionAvoidanceEnv()

print(cavoid.unsafe_spaces)
print('action_dim =', cavoid.action_space.shape[0])


def scale(x, b):
    x_min, x_max = b.low, b.high
    for i in range(2):
        x[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]
    return x


# Sampling from (reach - (safe + unsafe))
n_cuts = 80
x = torch.linspace(0, 1 - 1/n_cuts, n_cuts)
x = torch.cartesian_prod(x, x)
x = scale(x, cavoid.reach_space)
neg_safe = ~in_(cavoid.safe_space, x)
# for unsafe_space in cavoid.unsafe_spaces:
#     mask.logical_and_(~in_(unsafe_space, x))
x = x[neg_safe]

# sampling only from init
# x1 = scale(torch.rand(300, 2), cavoid.init_spaces[0])
# x2 = scale(torch.rand(300, 2), cavoid.init_spaces[1])
# x = torch.cat((x1, x2))


# SIMULATION SECTION
def simulate(x):
    safe = in2_(cavoid.unsafe_spaces, x)
    x = x[~safe]
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

    ax.scatter(x[:, 0], x[:, 1], s=1)
    for unsafe_space in cavoid.unsafe_spaces:
        unsafe = box2rect(unsafe_space, 'red', 'white')
        ax.add_patch(unsafe)
    for init_space in cavoid.init_spaces:
        init = box2rect(init_space, 'blue', 'white')
        ax.add_patch(init)
    safe = box2rect(cavoid.safe_space, 'green', 'white')
    ax.add_patch(safe)

    plt.pause(0.1)
    for i in range(30):
        print(f'=== step {i} ===')
        x = X[-1]
        x_nxt = cavoid.next(x, p(x)).detach()
        x_nxt = cavoid.add_noise(x_nxt)
        X.append(x_nxt)
        ax.scatter(x[:, 0], x[:, 1], s=1)
        reach = in_(cavoid.safe_space, x).sum()
        unsafe = (
            in_(cavoid.unsafe_spaces[0], x)
            + in_(cavoid.unsafe_spaces[1], x)
        ).sum()
        print(f'reach = {reach}')
        if unsafe:
            print(f'unsafe = {unsafe}')
        plt.draw()
        plt.pause(.1)

    plt.close(fig)
    plt.ioff()


def exp_v(x, V):
    def dupstate(x, dupfact=16):
        x1 = torch.unsqueeze(x, dim=0)
        x1 = x1.expand(dupfact, -1, -1)
        x1 = x1.flatten(end_dim=1)
        return x1
    x1 = dupstate(x)
    x1 = cavoid.add_noise(x1)
    v = V(x1)
    v1 = torch.cat(v.split(x.shape[0]), dim=1)
    v1 = v1.mean(dim=1)
    v1 = v1.unsqueeze(dim=1)
    return v1


p_ReachAvoid = 0.95


class Learner_ReachAvoid:
    def __init__(self, n_dims, models):
        # Abstraction A comes with a delta; so A is a non-deterministic
        # transition system.
        self.n_dims = n_dims
        self.V = models[0]

    def init_optimizer(self, lr):
        return torch.optim.SGD(
            list(self.V.parameters()) + list(p.parameters()), lr=lr)

    def loss(self, x):
        x_nxt = cavoid.next(x, p(x))

        # Any init state should be below 1
        barrier = 1 / (1 - p_ReachAvoid)
        init = in2_(cavoid.init_spaces, x).unsqueeze(dim=1)
        l_init = (torch.relu(self.V(x) - barrier) * init).mean()

        # Any unsafe state should be above barrier
        unsafe = in2_(cavoid.unsafe_spaces, x).unsqueeze(dim=1)
        l_unsafe = (
            torch.relu(barrier - self.V(x)) * unsafe).mean()
        # Comment the following line to
        # take the unsafe condition into account.
        # This condition doesn't need to be satisfied,
        # as the certificate can be constant
        # across any unsafe region.
        # l_unsafe = 0
        # Not unsafe, but maybe above barrier
        barrier_le = (self.V(x) <= barrier) * ~unsafe
        v_nxt = exp_v(x_nxt, self.V)
        l_dec = (
            torch.relu(
                v_nxt - self.V(x) + 1e-2) * barrier_le).mean()

        return l_init + l_unsafe + l_dec

    def chk(self, x):
        x_nxt = cavoid.next(x, p(x))

        barrier = 1 / (1 - p_ReachAvoid)
        init = in2_(cavoid.init_spaces, x).unsqueeze(dim=1)
        c_init = (self.V(x) <= barrier) * init + 1 * ~init

        unsafe = in2_(cavoid.unsafe_spaces, x).unsqueeze(dim=1)
        c_unsafe = (self.V(x) >= barrier) * unsafe + 1 * ~unsafe
        # Comment the following line to
        # take the unsafe condition into account.
        # This condition doesn't need to be satisfied,
        # as the certificate can be constant
        # across any unsafe region.
        # c_unsafe = torch.ones_like(c_init)
        barrier_le = (self.V(x) <= barrier) * ~unsafe
        v_nxt = exp_v(x_nxt, self.V)
        c_dec = (
            (self.V(x) >= v_nxt + 1e-3) * barrier_le
            + 1 * ~barrier_le)

        chk = c_init
        chk.logical_and_(c_unsafe)
        chk.logical_and_(c_dec)
        return chk.float().mean() * 100, chk

    def fit(self, S, n_epoch=512, batch_size=100, lr=1e-3, gamma=1.0):
        def training_step(s, optimizer):
            optimizer.zero_grad()
            loss = self.loss(s)
            chk, _ = self.chk(s)
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
                    + f'Chk={self.chk(S)[0]:12.6f}, '
                )
                # if self.chk(S) == 100.0:
                #     return

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


simulate(x)

learner = Learner_ReachAvoid(2, [nn_ReachAvoid(2)])
learner.fit(x, n_epoch=512, batch_size=120, lr=2e-3)

_, chk = learner.chk(x)
plt.scatter(x[:, 0], x[:, 1], s=2, c=chk[:x.shape[0], :])
plt.show()
plt.close()

v = learner.V(x).detach()
plt.scatter(x[:, 0], x[:, 1], s=2, c=v)
plt.show()
plt.close()
