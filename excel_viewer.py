import pandas as pd

class ExcelViewer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load(self, sheet_name=0):
        """Загрузить Excel-файл"""
        self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)

    def get_head(self, n=5):
        """Вернуть первые n строк"""
        return self.df.head(n) if self.df is not None else None

    def get_columns(self):
        """Вернуть список колонок"""
        return self.df.columns.tolist() if self.df is not None else None

    def filter(self, column, value):
        """Вернуть отфильтрованные данные"""
        if self.df is not None:
            return self.df[self.df[column] == value]
        return None

    def get_value(self, row, column):
        """Получить конкретное значение"""
        if self.df is not None:
            return self.df.iloc[row][column]
        return None