import pandas as pd
import numpy as np
import os


def read_csv_like(file_path, sep=None, has_header=False):
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if has_header:
            col = lines.pop(0).strip().split(sep=sep)
        for line in lines:
            li = line.strip().split(sep=sep)
            data.append(li)
    df = pd.DataFrame(data)
    df.columns = col
    return df


class DataDir:
    def __init__(self, *path):
        """path:: os.PathLike | str"""
        self.dir = os.path.abspath(os.path.join(*path))
        os.makedirs(self.dir, exist_ok=True)

    def load(self, file_name):
        full_path = os.path.join(self.dir, file_name)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            raise FileNotFoundError(f"数据集 {full_path} 不存在")
        return DataSet(source=full_path)


class DataSet:
    def __init__(self, source, hook=pd.read_csv, itype='', **hook_param):
        if itype == 'file' or (isinstance(source, (os.PathLike, str)) and os.path.exists(source)):
            self.full_df = hook(source, **hook_param)
            self._df = self.full_df
        elif itype == 'df' or isinstance(source, pd.DataFrame):
            self.full_df = source
            self._df = self.full_df
        elif itype == 'nd' or isinstance(source, np.ndarray):
            self.full_df = pd.DataFrame(source)
            self._df = self.full_df
        if source is not None:
            self._selected_columns = self._df  # 初始状态下选中的列是整个DataFrame
            self._filtered_df = self._df  # 初始状态下的行是整个DataFrame

        # 初始化迭代器相关属性
        self._iter_index = 0

    @property
    def df(self):
        # 当存在筛选列和行时，返回相应的 DataFrame
        now_df = self._filtered_df.loc[:, self._selected_columns.columns]
        self._df = self.full_df
        return now_df

    @property
    def nd(self):
        return self.df.values  # 根据最新的 df 生成 ndarray

    def select(self, *cols):
        """
        选择指定的列，支持按列名或列号（从1开始计数）。
        """
        selected_columns = []
        for col in cols:
            if isinstance(col, str):
                # 如果是字符串，则按列名选择
                selected_columns.append(self._df[col])
            elif isinstance(col, int):
                # 如果是整数，按列号选择（从1开始，减1以适配 iloc）
                selected_columns.append(self._df.iloc[:, col - 1])
        # 将选择的列存储起来，等待后续链式操作
        self._selected_columns = pd.concat(selected_columns, axis=1)
        return self  # 返回 self 以支持链式调用

    def where(self, **conditions):
        """
        根据条件过滤行，支持直接值和 lambda 表达式。
        """
        mask = pd.Series([True] * len(self._df))  # 初始化一个全为 True 的布尔掩码
        for column, condition in conditions.items():
            if callable(condition):
                # 如果 condition 是函数，使用 lambda 表达式
                mask = mask & self._df[column].apply(condition)
            else:
                # 如果 condition 是具体的值，直接进行匹配
                mask = mask & (self._df[column] == condition)
        # 更新过滤后的 DataFrame
        self._filtered_df = self._df[mask]
        return self  # 返回 self 以支持链式调用

    def save_csv(self, path):
        self._df.to_csv(path, index=False, encoding='utf-8-sig')

    def __iter__(self):
        """迭代器初始化"""
        self._iter_index = 0  # 重置迭代器
        return self

    def __next__(self):
        """获取下一个元素"""
        if self._iter_index < len(self._filtered_df):
            row = self._filtered_df.iloc[self._iter_index]  # 获取当前行
            self._iter_index += 1  # 移动到下一行
            return row
        else:
            raise StopIteration  # 结束迭代

    def apply_in_col(self, col_name_idx, func):
        self._df[col_name_idx] = self._df[col_name_idx].apply(func)
        return self

    def __repr__(self):
        """返回对象的正式字符串表示"""
        # return f"DataSet(rows={len(self._df)}, columns={len(self._df.columns)})"
        return self.__str__()

    def __str__(self):
        """返回对象的简洁字符串表示"""
        # 显示前几行数据和列信息
        return f"DataSet Columns: {list(self._df.columns)}\n\n-> Preview:\n{self._df.head()}"

    def __len__(self):
        return len(self._df)
