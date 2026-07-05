
### 2. `Common-LSG/core_gde_solver/operators.py`
```python
"""
LSG-GDE 基础微分算子库
构造三维空间微分算子 + 多层截面耦合算子，全部采用稀疏矩阵存储
严格对应：空间无限可分公理 + 层级截面几何范式
"""
import numpy as np
from scipy.sparse import diags, kron, eye, vstack


def build_3d_laplacian(nx: int, ny: int, nz: int, dx: float):
    """
    三维七点格式拉普拉斯稀疏矩阵（Dirichlet零边界）
    对应数学：∇²Φ，表征空间曲率/正维度凝聚强度
    """
    n = nx * ny * nz
    dx2 = dx ** 2

    main_diag = -6.0 / dx2 * np.ones(n)
    x_diag = 1.0 / dx2 * np.ones(n - 1)
    y_diag = 1.0 / dx2 * np.ones(n - nx)
    z_diag = 1.0 / dx2 * np.ones(n - nx * ny)

    # 消除跨边界虚假连接
    for i in range(1, nx):
        x_diag[i * ny * nz - 1] = 0

    return diags(
        [main_diag, x_diag, x_diag, y_diag, y_diag, z_diag, z_diag],
        [0, 1, -1, nx, -nx, nx*ny, -nx*ny],
        shape=(n, n),
        format="csr"
    )


def build_3d_gradient(nx: int, ny: int, nz: int, dx: float):
    """
    三维梯度稀疏矩阵，输出 3N×N，按 Gx/Gy/Gz 垂直堆叠
    对应数学：∇Φ，表征空间梯度/负维度弥散张力
    """
    n = nx * ny * nz
    dx_inv = 1.0 / (2 * dx)

    gx = diags([-dx_inv, dx_inv], [-1, 1], shape=(n, n), format="csr")
    gy = diags([-dx_inv, dx_inv], [-nx, nx], shape=(n, n), format="csr")
    gz = diags([-dx_inv, dx_inv], [-nx*ny, nx*ny], shape=(n, n), format="csr")

    return vstack([gx, gy, gz], format="csr")


def build_lambda_coupling(n_layers: int, n_spatial: int, dlambda: float):
    """
    LSG 层级截面耦合矩阵（二阶中心差分）
    对应数学：∂²/∂λ²，描述相邻尺度截面之间的场传递
    行和为0，严格满足空间本底守恒公理
    """
    main_diag = -2.0 / dlambda**2 * np.ones(n_layers)
    off_diag = 1.0 / dlambda**2 * np.ones(n_layers - 1)

    lambda_layer = diags(
        [main_diag, off_diag, off_diag],
        [0, 1, -1],
        shape=(n_layers, n_layers),
        format="csr"
    )
    # 扩展到全空间维度
    return kron(lambda_layer, eye(n_spatial, format="csr"), format="csr")


def build_global_operator(single_layer_op, n_layers: int):
    """
    将单截面算子扩展为全域分块对角算子
    对应LSG矩阵形式：每层截面独立计算，层间通过耦合矩阵关联
    """
    return kron(eye(n_layers, format="csr"), single_layer_op, format="csr")