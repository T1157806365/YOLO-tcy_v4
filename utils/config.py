"""
YOLO-tcy unified configuration utilities.

作用
----
1. 统一管理项目路径
2. 统一管理 YOLO11 / YOLO26 模型 YAML
3. 统一管理预训练权重
4. 统一管理 RGB / TIR / RGB-T 数据集 YAML
5. 后续 train_rgbt.py / val_rgbt.py / rgbt_model.py 都从这里取路径
6. 避免在多个 Python 文件中反复写绝对路径

项目根目录:
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml


# ============================================================
# 1. Project paths
# ============================================================

# utils/config.py -> utils -> YOLO-tcy
ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "configs"

DATASETS_DIR = CONFIG_DIR / "datasets"
MODELS_DIR = CONFIG_DIR / "models"
WEIGHTS_DIR = CONFIG_DIR / "weights"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
TRAIN_CONFIG_DIR = CONFIG_DIR / "train"

RUNS_DIR = ROOT / "runs"

PROJECT_PATHS = {
    "root": ROOT,
    "configs": CONFIG_DIR,
    "datasets": DATASETS_DIR,
    "models": MODELS_DIR,
    "weights": WEIGHTS_DIR,
    "experiments": EXPERIMENTS_DIR,
    "train": TRAIN_CONFIG_DIR,
    "runs": RUNS_DIR,
}


# ============================================================
# 2. Dataset registry
# ============================================================

# 这里只保存“配置文件路径”
# 真正图片路径仍然由各 dataset YAML 内部的 path/train/val/test 决定。
DATASET_REGISTRY = {
    # LRDD_v3
    "lrdd_rgb": DATASETS_DIR / "LRDD_v3-RGB.yaml",
    "lrdd_tir": DATASETS_DIR / "LRDD_v3-TIR.yaml",

    # 也允许写完整名称
    "LRDD_v3-RGB": DATASETS_DIR / "LRDD_v3-RGB.yaml",
    "LRDD_v3-TIR": DATASETS_DIR / "LRDD_v3-TIR.yaml",
}


# ============================================================
# 3. Supported YOLO models
# ============================================================

SUPPORTED_FAMILIES = {
    "yolo4",
    "yolo5",
    "yolo6",
    "yolo7",
    "yolo8",
    "yolo9",
    "yolo10",
    "yolo11",
    "yolo12",
    "yolo26",
}

SUPPORTED_SCALES = {
    "n",
    "s",
    "m",
    "l",
    "x",
}


# ============================================================
# 4. YAML utilities
# ============================================================

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取 YAML 文件。

    Parameters
    ----------
    path:
        YAML 文件路径。

    Returns
    -------
    dict
        YAML 内容。
    """

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"\nYAML 文件不存在:\n{path}\n"
        )

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"文件不是 YAML:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise TypeError(
            f"YAML 顶层必须是字典结构，但得到: {type(data)}"
        )

    return data


def save_yaml(
    data: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    保存 YAML。
    """

    path = Path(path).expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


# ============================================================
# 5. Dataset resolver
# ============================================================

def resolve_dataset_yaml(
    dataset: Union[str, Path],
) -> Path:
    """
    将数据集名称或 YAML 路径解析成完整路径。

    支持:

        lrdd_rgb

        lrdd_tir

        LRDD_v3-RGB

        LRDD_v3-TIR

    也支持:

        /xxx/xxx/my_dataset.yaml

    Parameters
    ----------
    dataset:
        registry 名称或 YAML 路径。

    Returns
    -------
    Path
    """

    dataset_str = str(dataset)

    # --------------------------------------------------------
    # 1. registry 名称
    # --------------------------------------------------------
    if dataset_str in DATASET_REGISTRY:
        path = DATASET_REGISTRY[dataset_str]

    else:
        # ----------------------------------------------------
        # 2. 用户直接传完整路径
        # ----------------------------------------------------
        candidate = Path(dataset_str).expanduser()

        if candidate.exists():
            path = candidate

        else:
            # ------------------------------------------------
            # 3. 尝试 configs/datasets/
            # ------------------------------------------------
            candidate = DATASETS_DIR / dataset_str

            if candidate.exists():
                path = candidate

            elif not candidate.suffix:
                candidate_yaml = candidate.with_suffix(".yaml")

                if candidate_yaml.exists():
                    path = candidate_yaml

                else:
                    raise FileNotFoundError(
                        "\n找不到数据集 YAML。\n"
                        f"输入: {dataset}\n\n"
                        f"Registry:\n"
                        f"{list(DATASET_REGISTRY.keys())}\n\n"
                        f"默认目录:\n"
                        f"{DATASETS_DIR}\n"
                    )
            else:
                raise FileNotFoundError(
                    "\n找不到数据集 YAML。\n"
                    f"输入: {dataset}\n"
                    f"默认目录: {DATASETS_DIR}\n"
                )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"数据集 YAML 不存在:\n{path}"
        )

    return path


# ============================================================
# 6. Dataset information
# ============================================================

def get_dataset_info(
    dataset: Union[str, Path],
) -> Dict[str, Any]:
    """
    读取数据集 YAML 并返回信息。

    返回中额外增加:
        yaml_file
    """

    yaml_path = resolve_dataset_yaml(dataset)

    data = load_yaml(yaml_path)

    data["yaml_file"] = str(yaml_path)

    return data


def check_dataset_pair(
    rgb_data: Union[str, Path],
    tir_data: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    检查 RGB 与 TIR 两个 YAML 是否存在，
    并检查类别数/类别名称是否兼容。

    注意
    ----
    此函数暂时只检查 dataset-level information。

    真正的:
        RGB image <-> TIR image

    一一配对检查将在 rgbt_dataset.py 中完成。
    """

    rgb_yaml = resolve_dataset_yaml(rgb_data)
    tir_yaml = resolve_dataset_yaml(tir_data)

    rgb_cfg = load_yaml(rgb_yaml)
    tir_cfg = load_yaml(tir_yaml)

    # --------------------------------------------------------
    # 检查类别名称
    # --------------------------------------------------------
    rgb_names = rgb_cfg.get("names")
    tir_names = tir_cfg.get("names")

    if rgb_names is not None and tir_names is not None:
        if rgb_names != tir_names:
            print(
                "\n"
                "WARNING: RGB 与 TIR 的 names 不完全一致。\n"
                f"RGB names: {rgb_names}\n"
                f"TIR names: {tir_names}\n"
            )

    # --------------------------------------------------------
    # 检查类别数
    # --------------------------------------------------------
    rgb_nc = rgb_cfg.get("nc")
    tir_nc = tir_cfg.get("nc")

    if (
        rgb_nc is not None
        and tir_nc is not None
        and int(rgb_nc) != int(tir_nc)
    ):
        raise ValueError(
            "\nRGB/TIR 类别数不一致:\n"
            f"RGB nc = {rgb_nc}\n"
            f"TIR nc = {tir_nc}\n"
        )

    return rgb_yaml, tir_yaml


# ============================================================
# 7. Model-name parser
# ============================================================

def parse_model_name(model_name: str):
    """
    Parse names such as:
        yolo5n, yolo5s, yolo5m
        yolo8n, yolo8s, yolo8m
        yolo11n, yolo11s, yolo11m
        yolo12n, yolo12s, yolo12m
        yolo26n, yolo26s, yolo26m
        yolo26m-p2

    Notes
    -----
    ``-p2`` is treated as an architecture suffix, not a new model scale.
    Therefore:
        yolo26m-p2 -> family="yolo26", scale="m"

    Returns
    -------
    family : str
        e.g. "yolo5", "yolo11", "yolo26"
    scale : str
        e.g. "n", "s", "m"
    """

    name = str(model_name).strip().lower()

    match = re.fullmatch(
        r"(yolo(?:5|6|7|8|9|10|11|12|26))([nsm])(u)?(?:-p2)?",
        name,
    )

    if match is None:
        raise ValueError(
            "\n不支持的模型名称:\n"
            f"{model_name}\n\n"
            "当前支持:\n"
            "  yolo5n / yolo5s / yolo5m\n"
            "  yolo6n / yolo6s / yolo6m\n"
            "  yolo7n / yolo7s / yolo7m\n"
            "  yolo8n / yolo8s / yolo8m\n"
            "  yolo9n / yolo9s / yolo9m\n"
            "  yolo10n / yolo10s / yolo10m\n"
            "  yolo11n / yolo11s / yolo11m\n"
            "  yolo12n / yolo12s / yolo12m\n"
            "  yolo26n / yolo26s / yolo26m / yolo26m-p2\n"
        )

    family = match.group(1)
    scale = match.group(2)

    return family, scale


# ============================================================
# 8. Model YAML resolver
# ============================================================

def resolve_model_yaml(
    model_name: str,
) -> Path:
    """
    根据模型名称找到网络 YAML。

    例如:

        yolo26n

    ->

        configs/models/yolo26/yolo26n.yaml
    """

    family, _ = parse_model_name(model_name)

    name = str(model_name).strip().lower()

    path = (
        MODELS_DIR
        / family
        / f"{name}.yaml"
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            "\n找不到模型 YAML:\n"
            f"{path}\n\n"
            "请确认你已经把官方模型 YAML 放在:\n"
            f"{MODELS_DIR}/{family}/\n"
        )

    return path


# ============================================================
# 9. Pretrained weight resolver
# ============================================================

def resolve_pretrained_weight(
    model_name: str,
) -> Path:
    """
    根据模型名称找到对应预训练权重。

    普通模型:
        yolo26m
        -> configs/weights/yolo26/yolo26m.pt

    P2 结构模型:
        yolo26m-p2
        -> configs/weights/yolo26/yolo26m.pt

    说明:
        ``-p2`` 只改变网络 YAML/检测头结构；
        预训练权重仍继承对应的标准 YOLO 基础模型。
        新增的 P2 分支由当前模型自行初始化。
    """

    family, _ = parse_model_name(model_name)

    name = str(model_name).strip().lower()

    # P2 YAML 没有单独的官方预训练权重。
    # yolo26m-p2 使用标准 yolo26m.pt 初始化可匹配的层。
    weight_name = re.sub(r"-p2$", "", name)

    path = (
        WEIGHTS_DIR
        / family
        / f"{weight_name}.pt"
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            "\n找不到预训练权重:\n"
            f"{path}\n\n"
            "请检查你的预训练模型是否位于:\n"
            f"{WEIGHTS_DIR}/{family}/\n"
        )

    # 防止下载成 HTML、LFS pointer 或空文件
    size_mb = (
        path.stat().st_size
        / 1024
        / 1024
    )

    if size_mb < 1.0:
        raise RuntimeError(
            "\n预训练权重文件大小异常:\n"
            f"{path}\n"
            f"文件大小: {size_mb:.3f} MB\n"
        )

    return path


# ============================================================
# 10. Unified model resolver
# ============================================================

def resolve_model(
    model_name: str,
    pretrained: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    一次获得模型 YAML + pretrained weight。

    Example
    -------
    >>> info = resolve_model("yolo26n")

    返回:
        {
            "name": "yolo26n",
            "family": "yolo26",
            "scale": "n",
            "yaml": Path(...),
            "weights": Path(...)
        }
    """

    family, scale = parse_model_name(
        model_name
    )

    model_yaml = resolve_model_yaml(
        model_name
    )

    weights = None

    if pretrained:
        weights = resolve_pretrained_weight(
            model_name
        )

    return {
        "name": model_name,
        "family": family,
        "scale": scale,
        "yaml": model_yaml,
        "weights": weights,
    }


# ============================================================
# 11. Image-size parser
# ============================================================

def parse_imgsz(
    imgsz: Union[
        int,
        str,
        Tuple[int, int],
        list,
    ]
):
    """
    支持:

        640

        "640"

        "640,640"

        "512,640"

        [512,640]

        (512,640)
    """

    if isinstance(imgsz, int):
        return imgsz

    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0])

        if len(imgsz) == 2:
            return [
                int(imgsz[0]),
                int(imgsz[1]),
            ]

        raise ValueError(
            f"imgsz 长度错误: {imgsz}"
        )

    if isinstance(imgsz, str):

        imgsz = imgsz.strip()

        if "," not in imgsz:
            return int(imgsz)

        values = [
            int(v.strip())
            for v in imgsz.split(",")
        ]

        if len(values) == 1:
            return values[0]

        if len(values) == 2:
            return values

        raise ValueError(
            f"无法解析 imgsz: {imgsz}"
        )

    raise TypeError(
        f"不支持的 imgsz 类型: {type(imgsz)}"
    )


# ============================================================
# 12. Experiment output
# ============================================================

def make_run_name(
    model_name: str,
    dataset_name: str,
    fusion: str,
    imgsz: Union[int, list, tuple],
    seed: int = 0,
) -> str:
    """
    自动生成实验名称。

    示例:

        yolo26n_LRDDv3_rgbt_early_640_seed0
    """

    if isinstance(imgsz, (list, tuple)):
        imgsz_str = "x".join(
            str(x)
            for x in imgsz
        )
    else:
        imgsz_str = str(imgsz)

    dataset_name = (
        dataset_name
        .replace(" ", "")
        .replace("/", "-")
    )

    return (
        f"{model_name}_"
        f"{dataset_name}_"
        f"{fusion}_"
        f"{imgsz_str}_"
        f"seed{seed}"
    )


# ============================================================
# 13. Directory checks
# ============================================================

def ensure_project_dirs() -> None:
    """
    创建需要自动生成内容的目录。

    不会创建模型/数据文件。
    """

    for path in [
        DATASETS_DIR,
        MODELS_DIR,
        WEIGHTS_DIR,
        EXPERIMENTS_DIR,
        TRAIN_CONFIG_DIR,
        RUNS_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 14. Print current project configuration
# ============================================================

def print_project_info() -> None:
    """
    打印项目目录。
    """

    print(
        "\n"
        "============================================================\n"
        "YOLO-tcy Project Configuration\n"
        "============================================================"
    )

    for key, value in PROJECT_PATHS.items():
        print(
            f"{key:<12}: {value}"
        )

    print(
        "============================================================\n"
    )


# ============================================================
# 15. Self test
# ============================================================

if __name__ == "__main__":

    print_project_info()

    print(
        "Checking LRDD_v3 dataset YAMLs..."
    )

    rgb_yaml, tir_yaml = check_dataset_pair(
        "lrdd_rgb",
        "lrdd_tir",
    )

    print(
        f"[OK] RGB YAML : {rgb_yaml}"
    )

    print(
        f"[OK] TIR YAML : {tir_yaml}"
    )

    print(
        "\nChecking YOLO pretrained models..."
    )

    for name in [
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo26n",
        "yolo26s",
        "yolo26m",
    ]:
        try:
            info = resolve_model(
                name,
                pretrained=True,
            )

            size_mb = (
                info["weights"]
                .stat()
                .st_size
                / 1024
                / 1024
            )

            print(
                f"[OK] "
                f"{name:<8} "
                f"YAML={info['yaml'].name:<15} "
                f"weight={size_mb:.2f} MB"
            )

        except Exception as e:
            print(
                f"[ERROR] {name}: {e}"
            )