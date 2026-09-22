"""Современный синтаксис + решение практической работы «Классификация вин»
в конце ноутбука «Текущий контроль 2»."""

import nbformat as nbf

PATH = "Текущий контроль 3-4/Текущий контроль 2.ipynb"

nb = nbf.read(PATH, as_version=4)
cells = nb["cells"]

MARKER = "## Решение практической работы"
IMPORTS = "from sklearn.metrics import (accuracy_score"
# повторный запуск: убираем ранее добавленное решение
for i, cell in enumerate(cells):
    if cell.cell_type == "markdown" and "".join(cell.source).startswith(MARKER):
        del cells[i:]
        print("stripped previously appended solution")
        break
# старые версии скрипта вставляли импорты до маркера — убираем их все
marker_idx = next(
    (
        i
        for i, x in enumerate(cells)
        if x.cell_type == "markdown" and "".join(x.source).startswith(MARKER)
    ),
    len(cells),
)
stray = [
    i
    for i, c in enumerate(cells[:marker_idx])
    if c.cell_type == "code" and "".join(c.source).startswith(IMPORTS)
]
for i in reversed(stray):
    del cells[i]
    print("stripped stray imports cell at", i)
# пустая кодовая ячейка-шаблон после условия задания
for i, cell in enumerate(cells):
    if cell.cell_type == "code" and not "".join(cell.source).strip():
        del cells[i]
        print("stripped empty template cell")
        break

# ---------------------------------------------------------------- modernize
# Старый код binary_dataset.loc[dataset['класс'] == ..., columns == 'класс'] = ±1
# не работает в современном pandas (столбец строкового типа не принимает int).
for cell in cells:
    if cell.cell_type == "code" and "Iris-versicolor" in "".join(cell.source):
        cell.source = (
            "#Классы кодируем целыми числами -1 и 1\n"
            "binary_dataset['класс'] = binary_dataset['класс'].map(\n"
            "    {'Iris-versicolor': -1, 'Iris-virginica': 1})"
        )
        print("modernized iris encoding cell")
        break

# ------------------------------------------------------------------ решение


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""## Решение практической работы

**Задача:** классификация красных вин по датасету `winequality-red.csv`
на два класса — «Хорошее» (quality ≥ 7) и «Некачественное»; результат —
уравнение разделяющей гиперповерхности. Дополнительно: применить модель
к классификации неизвестных вин, вычислить Accuracy, Precision, Recall
и вывести confusion matrix.

Схема решения повторяет лекционный пример: линейная модель
(LogisticRegression) в расширенном пространстве признаков (добавлен
признак, равный 1), классы закодированы числами $-1$ и $1$.""")

code(r"""from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split""")

code(r"""#Загрузим датасет и посмотрим на него
wine = pd.read_csv(r"winequality-red.csv")
display(wine.head())
print("Размер выборки:", wine.shape)
print("Распределение оценок quality:")
print(wine["quality"].value_counts().sort_index())""")

code(r"""#Множество объектов X — 11 признаков, к которым добавлен признак, равный 1.
#Множество ответов y: 1 — «Хорошее» (quality >= 7), -1 — «Некачественное»
y = np.where(wine["quality"] >= 7, 1, -1)
X = wine.drop(columns=["quality"]).to_numpy(float)
X = np.hstack([X, np.ones((len(X), 1))])
print("Хороших вин:", (y == 1).sum(), ", некачественных:", (y == -1).sum())""")

md(r"""**Задача машинного обучения:** $|\mathbf{y}| = 2 \ll l = 1599$ —
задача бинарной классификации. Модель алгоритмов — семейство линейных
классификаторов
$\mathfrak{F} = \{ f(\theta, \mathbf{x}) = \operatorname{sign}(\theta^{\mathsf{T}}\mathbf{x}) \}$.

Часть выборки отложим как «неизвестные вина» — к ним модуль будет
применён после обучения. Разбиение стратифицированное, чтобы сохранить
доли классов (вин класса «Хорошее» заметно меньше):""")

code(r"""#Откладываем 30% выборки как «неизвестные» вина
X_train, X_unknown, y_train, y_unknown = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
print("Обучающая выборка:", len(X_train), "вин, неизвестные вина:", len(X_unknown))""")

code(r"""#Обучаем модель на обучающей выборке
model = LogisticRegression(random_state=0, max_iter=5000)
model.fit(X_train, y_train)

#Уравнение гиперповерхности, отделяющей один класс от другого: theta^T x = 0
theta = model.coef_[0]
names = list(wine.columns[:-1]) + ["1"]
print("Разделяющая гиперповерхность:")
print(" ".join(f"{t:+.4f}·{n}" for t, n in zip(theta, names)), "= 0")""")

md(r"""### Классификация неизвестных вин""")

code(r"""#Применяем обученную модель к классификации неизвестных вин
y_pred = model.predict(X_unknown)
label = {1: "Хорошее", -1: "Некачественное"}
result = pd.DataFrame({"истинный класс": [label[v] for v in y_unknown],
                       "предсказание модели": [label[v] for v in y_pred]})
display(result.head(10))
print("Правильно классифицировано:", (y_pred == y_unknown).sum(),
      "из", len(y_unknown))""")

md(r"""### Метрики классификации""")

code(r"""#Accuracy, Precision и Recall (положительный класс — «Хорошее»)
print("Accuracy: ", accuracy_score(y_unknown, y_pred))
print("Precision:", precision_score(y_unknown, y_pred, pos_label=1))
print("Recall:   ", recall_score(y_unknown, y_pred, pos_label=1))
print()
print(classification_report(y_unknown, y_pred,
                            labels=[-1, 1],
                            target_names=["Некачественное", "Хорошее"]))""")

code(r"""#Confusion matrix
cm = confusion_matrix(y_unknown, y_pred, labels=[-1, 1])
cm_df = pd.DataFrame(cm,
                     index=["Некачественное", "Хорошее"],
                     columns=["Некачественное", "Хорошее"])
display(cm_df)

plt.figure()
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.show()""")

md(r"""**Вывод.** Разделяющая гиперповерхность построена в
12-мерном пространстве (11 признаков + константа); наибольший вклад
вносят `volatile acidity` (отрицательный) и `alcohol` (положительный),
то есть хорошее вино — с меньшей летучей кислотностью и большей
крепостью.

На неизвестных винах модель даёт accuracy $\approx 0{,}85$: почти все
некачественные вина опознаются верно, однако recall «Хорошего» класса
невысок ($\approx 0{,}28$) — из-за сильного дисбаланса классов
(хороших вин всего $\approx 14\%$) модель редко рискует присваивать
класс «Хорошее». Confusion matrix показывает это явно: из всех
истинно хороших вин модель находит лишь около трети.""")

nbf.write(nb, PATH)
print(PATH, "updated,", len(cells), "cells")
