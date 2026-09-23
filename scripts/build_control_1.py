"""Дополнение ноутбука «Текущий контроль 1» решением задания 2
(парная линейная регрессия для гиперспектра кукурузы)."""

import nbformat as nbf

PATH = "Текущий контроль 1-2/Текущий контроль 1.ipynb"

nb = nbf.read(PATH, as_version=4)
cells = nb["cells"]

MARKER = "## Решение задания"
# повторный запуск: убираем ранее добавленное решение
for i, cell in enumerate(cells):
    if cell.cell_type == "markdown" and "".join(cell.source).startswith(MARKER):
        del cells[i:]
        print("stripped previously appended solution")
        break


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(r"""## Решение задания""")

# ------------------------------------------------------------------ данные
code(r"""#Загрузим данные: разделитель столбцов ';', десятичный разделитель ','
gipers = pd.read_csv(r"Гиперспектр кукурузы.csv", sep=";", decimal=",")
display(gipers.head())
x = gipers["wavelength"].to_numpy(float)  # длина волны, нм
Y = gipers["Spectr"].to_numpy(float)      # интенсивность спектра
print("Точек:", len(x), ", длина волны: от", x.min(), "до", x.max(), "нм")

#Изобразим обучающую выборку на графике
plt.figure()
plt.scatter(x, Y, s=10)
plt.xlabel("Длина волны, нм")
plt.ylabel("Spectr")
plt.show()""")

# ------------------------------------------------------- 1. аналитически
md(r"""### 1. Оценки коэффициентов аналитическим путём

Используем формулы метода наименьших квадратов, как в примере выше.""")

code(r"""#Вычислим оценки коэффициентов парной линейной регрессии по формулам
a1 = ((x - x.mean())*(Y - Y.mean())).mean()/((x - x.mean())**2).mean()
a0 = Y.mean() - a1*x.mean()
print("Модель линейной регрессии: Y^ =", a0, " +", a1, "* x")

#Изобразим на графике обучающую выборку и аналитическую модель
x_space = np.linspace(x.min(), x.max(), 100)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x, Y, s=10, label="обучающая выборка")
ax.plot(x_space, a0 + a1*x_space, 'r', label="аналитическая модель")
ax.legend()
plt.show()""")

# ------------------------------------------------- 2. градиентный спуск
md(r"""### 2. Оценки коэффициентов методом градиентного спуска

Значения $x$ (длина волны, $\approx 400 \ldots 1000$ нм) в тысячи раз
больше значений $Y$, поэтому задача минимизации плохо обусловлена:
шаг градиентного спуска, подходящий для $a_1$, «разрушает» $a_0$,
а маленький шаг делает спуск неприемлемо медленным. Проверим это:""")

code(r"""#Попробуем тот же класс SimpleRegression на исходном масштабе данных
regr_raw = SimpleRegression()
steps_raw, errors_raw = regr_raw.fit(x, Y, alpha=1e-6, epsylon=0.0177, max_steps=10)
print("MSE по шагам:", errors_raw)  # MSE растёт взрывным образом""")

md(r"""Стандартный приём в такой ситуации — **стандартизация признака**:
$z = \dfrac{x - \bar x}{\sigma_x}$. Линейная модель от этого не меняется
($Y = c_0 + c_1 z$ — тот же семейство прямых), но градиентный спуск
сходится. После обучения коэффициенты переводим обратно в исходный
масштаб: $a_1 = \dfrac{c_1}{\sigma_x}$, $a_0 = c_0 - \dfrac{c_1 \bar x}{\sigma_x}$.""")

code(r"""#Стандартизуем x и запустим градиентный спуск (тот же alpha, что в примере выше)
mu, sd = x.mean(), x.std()
z = (x - mu)/sd
regr = SimpleRegression()
steps, errors = regr.fit(z, Y, alpha=0.001, epsylon=0.0177)
print("MSE после градиентного спуска: ", regr.MSE(z, Y))

#Переводим коэффициенты в исходный масштаб: Y = c0 + c1*z = a0 + a1*x
a0_gd = regr.a0 - regr.a1*mu/sd
a1_gd = regr.a1/sd
print("Модель линейной регрессии: Y^ =", a0_gd, " +", a1_gd, "* x")

#Кривая обучения — изменение MSE в процессе градиентного спуска
plt.figure()
plt.plot(steps, errors)
plt.xlabel("шаг")
plt.ylabel("MSE")
plt.show()""")

# --------------------------------------------------- 3. качество моделей
md(r"""### 3. Оценка качества моделей

Сравним обучающую выборку с прогнозами обеих моделей на одном графике
и вычислим метрики ошибок MSE, MAE и MAPE.""")

code(r"""#Прогнозы обеих моделей на обучающей выборке
Y_pred_an = a0 + a1*x
Y_pred_gd = a0_gd + a1_gd*x

print("аналитическая модель: MSE =", ((Y - Y_pred_an)**2).mean(),
      " MAE =", abs(Y - Y_pred_an).mean(),
      " MAPE =", (abs((Y - Y_pred_an)/Y)).mean())
print("градиентный спуск:    MSE =", ((Y - Y_pred_gd)**2).mean(),
      " MAE =", abs(Y - Y_pred_gd).mean(),
      " MAPE =", (abs((Y - Y_pred_gd)/Y)).mean())

#Сравним на графике обучающую выборку и прогнозы обеих моделей
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x, Y, s=10, label="обучающая выборка")
ax.plot(x_space, a0 + a1*x_space, 'r', label="аналитическая модель")
ax.plot(x_space, a0_gd + a1_gd*x_space, 'g--', label="градиентный спуск")
ax.legend()
plt.show()""")

md(r"""**Вывод.** Оценки коэффициентов, полученные аналитически
($a_0 \approx -0{,}1834$, $a_1 \approx 6{,}68 \cdot 10^{-4}$) и методом
градиентного спуска после стандартизации признака, практически совпадают,
прямые на графике неотличимы (MSE $\approx 0{,}0177$).

Качество прямолинейной модели для этих данных невысокое: зависимость
Spectr от длины волны существенно нелинейна, поэтому ошибки прогноза
велики (MAE $\approx 0{,}10$ при диапазоне $Y$ от $0{,}02$ до $0{,}62$).
MAPE получается особенно большим, так как в области малых значений $Y$
(короткие волны) относительная ошибка неконтролируемо растёт.""")

# -------------------------------------------------- нелинейная аппроксимация
md(r"""### Нелинейная аппроксимация: полиномы степеней 2–5

Прямая плохо описывает нелинейную зависимость Spectr от длины волны.
Естественное нелинейное обобщение — полиномиальная модель

$$Y = b_0 + b_1 x + b_2 x^2 + \ldots + b_k x^k.$$

Она **линейна по параметрам** $b_j$, поэтому вся техника предыдущего
раздела применима без изменений: матрица объекты-признаки составляется
из степеней $x$, оценки ищутся либо решением нормальных уравнений
$(\Phi^{\mathsf{T}}\Phi)\,\theta = \Phi^{\mathsf{T}}Y$, либо градиентным
спуском по стандартизованным степеням $z = (x - \bar x)/\sigma$
(без стандартизации спуск расходится).""")

code('''#Единые функции для полиномиальной аппроксимации любой степени
Pol = np.polynomial.Polynomial


def poly_fit(x, Y, degree, method="analytic", alpha=1e-4, epsylon=1e-4, max_steps=5000):
    """Полином Y ~ b0 + b1*x + ... + b_degree*x^degree (линейный по параметрам).

    method="analytic" — решение нормальных уравнений (Ф^T Ф) t = Ф^T Y;
    method="gd" — градиентный спуск по стандартизованным степеням.
    Возвращает (коэффициенты в исходном масштабе x, шаги, ошибки)."""
    mu, sd = x.mean(), x.std()
    z = (x - mu)/sd
    Phi = np.column_stack([z**k for k in range(degree + 1)])
    if method == "analytic":
        cz, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
        steps = errors = None
    else:
        cz = np.zeros(degree + 1)
        steps, errors = [], []
        for step in range(max_steps):
            pred = Phi @ cz
            cz -= alpha * (-2 * Phi.T @ (Y - pred))
            new_error = ((Y - pred)**2).mean()
            steps.append(step + 1)
            errors.append(new_error)
            if new_error < epsylon:
                break
    #перевод коэффициентов из z в исходный масштаб x:
    #p((x - mu)/sd) как многочлен от x
    coef = Pol(cz)(Pol([-mu/sd, 1/sd])).coef
    return coef, steps, errors


def poly_metrics(Y, coef, x):
    """MSE, MAE и MAPE полиномиальной модели на выборке (x, Y)."""
    pred = Pol(coef)(x)
    return {"MSE": ((Y - pred)**2).mean(),
            "MAE": abs(Y - pred).mean(),
            "MAPE": (abs((Y - pred)/Y)).mean()}''')

code(r"""#Обучим полиномы степеней 1-5 и изобразим их на одном графике
degrees = [1, 2, 3, 4, 5]
fits = {}
rows = []
x_space = np.linspace(x.min(), x.max(), 300)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x, Y, s=10, label="обучающая выборка")
for d in degrees:
    coef, _, _ = poly_fit(x, Y, d)
    fits[d] = coef
    rows.append({"степень полинома": d, **poly_metrics(Y, coef, x)})
    ax.plot(x_space, Pol(coef)(x_space), label=f"полином {d}-й степени")
ax.legend()
plt.show()

#Сводная таблица метрик качества
display(pd.DataFrame(rows))""")

code(r"""#Проверка: тот же результат даёт градиентный спуск (method="gd")
d = 3
coef_gd, steps_gd, errors_gd = poly_fit(x, Y, d, method="gd", alpha=1e-4)
print("MSE после градиентного спуска:", errors_gd[-1])
print("совпадает с аналитическим решением:",
      np.allclose(coef_gd, fits[d], atol=1e-3))

#Кривая обучения — изменение MSE в процессе градиентного спуска
plt.figure()
plt.plot(steps_gd, errors_gd)
plt.xlabel("шаг")
plt.ylabel("MSE")
plt.show()""")

md(r"""**Вывод.** MSE монотонно убывает с ростом степени:
$0{,}0177$ (прямая) $\to 0{,}0152$ (ст. 2) $\to 0{,}0122$ (ст. 3)
$\to 0{,}0122$ (ст. 4) $\to 0{,}0069$ (ст. 5). Полином 5-й степени
первым сгибается в пик около 700–800 нм и заметно лучше описывает
форму спектра; степени 3 и 4 почти не отличаются.

MAE и MAPE ведут себя немонотонно: они чувствительны к области коротких
волн, где $Y$ близко к нулю и относительная ошибка велика. Дальнейший
рост степени уменьшил бы ошибку на обучающей выборке ещё сильнее, но
привёл бы к переобучению — полином начал бы подстраиваться под шум.""")

nbf.write(nb, PATH)
print(PATH, "updated,", len(cells), "cells")
