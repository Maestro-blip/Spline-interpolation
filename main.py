import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline


data =[
  {"elevation": 1663, "longitude": 7.7500, "latitude": 46.0167},
  {"elevation": 1740, "longitude": 7.7398, "latitude": 46.0126},
  {"elevation": 1830, "longitude": 7.7296, "latitude": 46.0085},
  {"elevation": 1940, "longitude": 7.7194, "latitude": 46.0044},
  {"elevation": 2060, "longitude": 7.7092, "latitude": 46.0003},
  {"elevation": 2200, "longitude": 7.6990, "latitude": 45.9962},
  {"elevation": 2350, "longitude": 7.6888, "latitude": 45.9921},
  {"elevation": 2550, "longitude": 7.6786, "latitude": 45.9880},
  {"elevation": 2900, "longitude": 7.6684, "latitude": 45.9839},
  {"elevation": 4181, "longitude": 7.6586, "latitude": 45.9763}
]

lat = np.array([p["latitude"] for p in data])
lon = np.array([p["longitude"] for p in data])
elev = np.array([p["elevation"] for p in data])

# Перетворення координат у метри
# 111320 м — приблизна відстань по меридіану (схід-захід) для 1° широти. Тобто зміна широти на 1° ≈ 111.32 км.
lat_m = (lat - lat[0]) * 111320
lon_m = (lon - lon[0]) * 111320 * np.cos(np.radians(lat[0]))

# Кумулятивна відстань
dist = np.zeros(len(lat))
for i in range(1, len(lat)):
    dx = lon_m[i] - lon_m[i-1]
    dy = lat_m[i] - lat_m[i-1]
    dist[i] = dist[i-1] + np.sqrt(dx**2 + dy**2)

# -----------------------------
# 2. Кубічний сплайн з нуля
# -----------------------------
n = len(dist) - 1
h = np.diff(dist)

# Матриця для трьохдіагональної системи
alpha = np.zeros(n-1)
beta = np.zeros(n-1)
gamma = np.zeros(n-1)
d = np.zeros(n-1)

for i in range(1, n):
    alpha[i-1] = h[i-1]
    beta[i-1] = 2*(h[i-1]+h[i])
    gamma[i-1] = h[i]
    d[i-1] = 3*((elev[i+1]-elev[i])/h[i] - (elev[i]-elev[i-1])/h[i-1])

# -----------------------------
# 3. Метод прогонки для трьохдіагональної матриці
# -----------------------------
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n)
    d_star = np.zeros(n)
    x = np.zeros(n)

    c_star[0] = c[0]/b[0]
    d_star[0] = d[0]/b[0]

    for i in range(1, n):
        m = b[i] - a[i]*c_star[i-1]
        c_star[i] = c[i]/m if i < n-1 else 0
        d_star[i] = (d[i] - a[i]*d_star[i-1])/m

    x[-1] = d_star[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i]*x[i+1]

    return x

# -----------------------------
# 4. Знаходження коефіцієнтів c
# -----------------------------
c_internal = thomas_algorithm(alpha, beta, gamma, d)
c = np.zeros(n+1)
c[1:n] = c_internal

# -----------------------------
# 5. Знаходження коефіцієнтів a, b, d
# -----------------------------
a_coef = elev[:-1]
b_coef = (elev[1:] - elev[:-1])/h - h*(2*c[:-1] + c[1:])/3
d_coef = (c[1:] - c[:-1])/(3*h)
print("before main")

for i in range(n):
    print(f"Інтервал {i}: x = [{dist[i]:.2f}, {dist[i+1]:.2f}]")
    print(f"  a[{i}] = {a_coef[i]:.4f}")
    print(f"  b[{i}] = {b_coef[i]:.4f}")
    print(f"  c[{i}] = {c[i]:.4f}")
    print(f"  d[{i}] = {d_coef[i]:.6f}")
    print("-"*40)

# -----------------------------
# 6. Табуляція та похибка
# -----------------------------
N = 500
dist_fine = np.linspace(dist[0], dist[-1], N)
spline_values = np.zeros(N)

for i in range(n):
    mask = (dist_fine >= dist[i]) & (dist_fine <= dist[i+1])
    xi = dist_fine[mask]
    dx_i = xi - dist[i]
    spline_values[mask] = a_coef[i] + b_coef[i]*dx_i + c[i]*dx_i**2 + d_coef[i]*dx_i**3

# Похибка (за умови що “істинні” y = cs(dist_fine) для порівняння)

cs_true = CubicSpline(dist, elev)
y_true = cs_true(dist_fine)


# -----------------------------
# 7. Кумулятивна енергія та градієнт
# -----------------------------
dh = np.diff(spline_values)
dx = np.diff(dist_fine)
grade_percent = dh/dx*100

m = 70
g = 9.81
dE = m * g * np.maximum(dh, 0)
E_cumulative = np.cumsum(dE)
def main():


    error = y_true - spline_values
    # -----------------------------
    # Таблиця результатів
    # -----------------------------
    table = pd.DataFrame({
        "Distance_m": dist_fine[1:],
        "Elevation_m": spline_values[1:],
        "Grade_%": grade_percent,
        "Cumulative_Energy_J": E_cumulative,
        "Spline_Error_m": error[1:]
    })

    pd.set_option("display.max_rows", None)
    print(table)

    # -----------------------------
    # Графіки
    # -----------------------------

    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    plt.plot(dist_fine, y_true, label="Справжня функція", color='blue')
    plt.plot(dist_fine, spline_values, '--', label="Кубічний сплайн", color='orange')
    plt.ylabel("Висота, м")
    plt.title("Профіль висоти та сплайн для маршруту Церматт → Маттергорн")
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(dist_fine[1:], grade_percent, color='green')
    plt.ylabel("Градієнт, %")
    plt.title("Градієнт траси для маршруту Церматт → Маттергорн")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(dist_fine[1:], E_cumulative/1000, color='red')
    plt.ylabel("Кумулятивна енергія, кДж")
    plt.xlabel("Відстань, м")
    plt.title("Кумулятивні енергетичні витрати на підйом для маршруту Церматт → Маттергорн")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    error = y_true - spline_values
    MAE = np.mean(np.abs(error))
    print(f"Середня абсолютна похибка сплайна: {MAE:.4f} м")
    MSE = np.mean(error**2)
    RMSE = np.sqrt(MSE)
    print(f"Корінь середньоквадратичної похибки: {RMSE:.4f} м")




if __name__ == "__main__":
     main()
