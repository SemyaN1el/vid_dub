import os
import gc
import shutil
import random
import logging
import numpy as np
import torch
from typing import NoReturn

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed зафиксирован: {seed}")


def manage_directory(path: str, action: str = "create") -> None:
    """Создаёт или удаляет директорию."""
    if action == "create":
        os.makedirs(path, exist_ok=True)
        logger.info(f"Директория создана: {path}")
    elif action == "delete":
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"Директория удалена: {path}")
    else:
        raise ValueError(f"Неизвестное действие: {action}. Используйте 'create' или 'delete'.")


def create_path(directory: str, filename: str) -> str:
    """Объединяет директорию и имя файла в путь."""
    path = os.path.join(directory, filename)
    logger.debug(f"Путь создан: {path}")
    return path


def print_directory_tree(start_path: str = "./", max_depth: int = 4) -> None:
    """Рекурсивно выводит дерево директорий до заданной глубины."""
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, "").count(os.sep)
        if level > max_depth:
            continue
        indent = "│   " * (level - 1) + "├── " if level > 0 else ""
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{'│   ' * level}└── {f}")


def free_gpu_memory(*model_names: str, global_vars: dict = None) -> None:
    """
    Удаляет модели из памяти и освобождает кэш GPU.

    Параметры:
        *model_names: имена переменных моделей для удаления
        global_vars: словарь globals() из вызывающего модуля
    """
    if global_vars:
        for name in model_names:
            if name in global_vars:
                del global_vars[name]
                logger.info(f"Удалено из памяти: {name}")
            else:
                logger.warning(f"Переменная не найдена: {name}")

    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 2
    logger.info(f"GPU после очистки — занято: {allocated:.1f} MB, зарезервировано: {reserved:.1f} MB")


def normalize_path(path: str) -> str:
    """Нормализует путь: заменяет обратные слэши на прямые (для Windows)."""
    return path.replace("\\", "/")
