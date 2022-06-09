"""
NIMS에 적용되는 커스텀 transform을 정의하는 모듈입니다. 일반적인 transform은 torchvision.transforms에서
찾으실 수 있습니다. `train.py` 참고.

"""
__all__ = ["InterpolateAWS", "ToTensor", "ClassifyByThresholds", "UndersampleDry", "UndersampleGlobal"]

import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def IDW_interpolated(points, values, w, h, grid_x, grid_y, alpha_parameter, R_parameter):
    # 처음에 더한 1 다시 빼주기
    values -= 1
    # interpolated: 최종적으로 내보낼 array
    interpolated = np.zeros((w, h))
    # points에 해당하는 값을 미리 넣어줌
    for i in range(len(points)):
        interpolated[points[i][0], points[i][1]] = values[i]

    # all_list: aws_grid의 모든 좌표->list
    all_list = []

    # grid_x_1, grid_y_1: array를 한줄로 만든 형태
    grid_x_1 = grid_x.reshape(1, w * h)
    grid_y_1 = grid_y.reshape(1, w * h)
    for i in range(w * h):
        all_list.append([grid_x_1[0][i], grid_y_1[0][i]])
    # all_points: aws_grid의 모든 좌표들의 np.array
    all_points = np.array(all_list)

    # pfi(points for interpolate): AWS에서 측정값이 없는 points들의 np.array
    a1 = all_points
    a2 = points
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    pfi = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    n, _ = pfi.shape

    xs = points[:, 0]
    ys = points[:, 1]

    for i in range(n):
        x = pfi[i][0]
        y = pfi[i][1]
        distance = np.sqrt((xs - x) ** 2 + (ys - y) ** 2) ** (-alpha_parameter)
        # R_parameter보다 멀면 0으로 처리
        for j in range(len(distance)):
            if distance[j] < (R_parameter) ** (-alpha_parameter):
                distance[j] = 0
        # 주변에 AWS값이 있는 grid가 없는 경우 NULL값으로 처리
        if np.sum(distance) == 0:
            IDW = -9999
        else:
            IDW = np.sum(distance * values) / np.sum(distance)
            # 강수량이 0.5 단위로 나오도록 fitting
            IDW = round(IDW * 2) / 2
        interpolated[pfi[i][0], pfi[i][1]] = IDW

    return interpolated


from scipy import interpolate


def spline_interpolated(points, values, w, h, grid_x, grid_y):
    # 처음에 더한 1 다시 빼주기
    values -= 1
    # interpolated: 최종적으로 내보낼 array
    interpolated = np.zeros((w, h))
    # points에 해당하는 값을 미리 넣어줌
    for i in range(len(points)):
        interpolated[points[i][0], points[i][1]] = values[i]

    # all_list: aws_grid의 모든 좌표->list
    all_list = []

    # grid_x_1, grid_y_1: array를 한줄로 만든 형태
    grid_x_1 = grid_x.reshape(1, w * h)
    grid_y_1 = grid_y.reshape(1, w * h)
    for j in range(w * h):
        all_list.append([grid_x_1[0][j], grid_y_1[0][j]])
    # all_points: aws_grid의 모든 좌표들의 np.array
    all_points = np.array(all_list)

    # pfi(points for interpolate): AWS에서 측정값이 없는 points들의 np.array
    a1 = all_points
    a2 = points
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    pfi = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    n, _ = pfi.shape

    xs = points[:, 0]
    ys = points[:, 1]

    z = values
    f = interpolate.interp2d(xs, ys, z, kind='quintic')

    for i in range(n):
        x = pfi[i][0]
        y = pfi[i][1]
        interpolated[pfi[i][0], pfi[i][1]] = f(x, y)

    return interpolated


def GP_interpolated(points, values, w, h, grid_x, grid_y):
    # 처음에 더한 1 다시 빼주기
    values -= 1
    # interpolated: 최종적으로 내보낼 array
    interpolated = np.zeros((w, h))
    # points에 해당하는 값을 미리 넣어줌
    for i in range(len(points)):
        interpolated[points[i][0], points[i][1]] = values[i]

    # all_list: aws_grid의 모든 좌표->list
    all_list = []

    # grid_x_1, grid_y_1: array를 한줄로 만든 형태
    grid_x_1 = grid_x.reshape(1, w * h)
    grid_y_1 = grid_y.reshape(1, w * h)
    for i in range(w * h):
        all_list.append([grid_x_1[0][i], grid_y_1[0][i]])
    # all_points: aws_grid의 모든 좌표들의 np.array
    all_points = np.array(all_list)

    # pfi(points for interpolate): AWS에서 측정값이 없는 points들의 np.array
    a1 = all_points
    a2 = points
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    pfi = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    n, _ = pfi.shape
    # pfi_value: GP를 통해 얻어낸 point들의 강수량 np.array
    gpr = GaussianProcessRegressor().fit(points, values)
    pfi_value = gpr.predict(pfi, return_std=False, return_cov=False)

    # for i in range(n):
    #     #강수량이 0.5단위로 나오도록 fitting
    #     interpolated[pfi[i][0],pfi[i][1]] = round(pfi_value[i]*2)/2

    for i in range(n):
        interpolated[pfi[i][0], pfi[i][1]] = pfi_value[i]

    return interpolated


class InterpolateAWS:
    def __call__(self, aws_grid: torch.Tensor) -> torch.Tensor:
        """
        :param aws_grid: W x H 형태의 AWS 관측 데이터
        :return:
        """

        method = 1

        ##aws_grid에 1을 더해준 다음 0이 아닌 부분들에 대해서 고려를 끝낸 뒤 interpolated에서 다시 1을 빼주는 방법을 쓴다 -> sparse 를 사용하기 위해서
        aws_grid[aws_grid == -9999] = -1
        aws_grid += 1
        sparse = csr_matrix(aws_grid)  # hotfix, since csr_matrix considers 0 as null

        w, h = aws_grid.shape
        grid_x, grid_y = np.mgrid[0:w:1, 0:h:1]

        rows, cols = sparse.nonzero()
        rows, cols = rows.tolist(), cols.tolist()
        ##points: x가 row, y가 col인 AWS 실제값이 존재하는 grid들 좌표 / numpy array
        ##values: x,y에 있는 value / numpy array
        points = np.array(list(zip(rows, cols)))
        values = sparse.data

        if method == 1:
            # linear interpolation
            interpolated = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=-9998, rescale=False)
            interpolated = interpolated - 1  # shift back to -9999, [0, inf)
        if method == 2:
            # IDW interpolation
            alpha_parameter = 1.5
            R_parameter = 10
            interpolated = IDW_interpolated(points, values, w, h, grid_x, grid_y, alpha_parameter, R_parameter)
        if method == 3:
            # spline interpolation
            interpolated = spline_interpolated(points, values, w, h, grid_x, grid_y)
        if method == 4:
            # GP interpolation
            interpolated = GP_interpolated(points, values, w, h, grid_x, grid_y)

        # interpolated = aws_grid-1

        # # plotting
        # from PIL import Image
        # interpolated2 = np.flipud(interpolated)
        # pyplot.imshow(interpolated2,interpolation='none')
        # pyplot.savefig('spline.png')

        return torch.Tensor(interpolated)


class ClassifyByThresholds:
    """
    AWS에서 관측된 강수량을 threshold에 따라 클래스로 분류
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        :param grid: 강수 데이터를 담은 W x H 그리드. 누락된 값은 -9999로 처리되어 있어야 함.
        """
        thresholds = [0.0] + self.thresholds + [float('inf')]
        result = grid
        for i, start in enumerate(thresholds[:-1]):
            end = thresholds[i + 1]
            result = torch.where((start <= grid) & (grid < end), i * torch.ones(grid.shape), result)
        return result


class UndersampleDry:
    """
    Class label이 0인 경우(강수 없음) 중 일부만 임의로 선택하여 사용 (-9999로 누락 처리) (dry undersampling)
    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """
        :param label: class label 정보를 담은 W x H 그리드.
        """
        label[(label == 0) & (torch.rand_like(label) >= self.sampling_rate)] = -9999
        return label


class UndersampleGlobal:
    """
    모든 point 중 일부만 임의로 선택하여 사용 (global undersampling)
    """

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """
        :param label: class label 정보를 담은 W x H 그리드.
        """
        label[torch.rand_like(label) >= self.sampling_rate] = -9999
        return label


class ToTensor:
    def __call__(self, images):
        return torch.from_numpy(images)
