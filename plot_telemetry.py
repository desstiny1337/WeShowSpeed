import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("telemetry_log.csv")

# 1. Скорость и target_speed
plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["speedX"], label="speedX")
plt.plot(df["step"], df["target_speed"], label="target_speed")
plt.xlabel("Step")
plt.ylabel("Speed")
plt.title("Speed vs target speed")
plt.legend()
plt.grid(True)
plt.show()

# 2. Газ и тормоз
plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["accel"], label="accel")
plt.plot(df["step"], df["brake"], label="brake")
plt.xlabel("Step")
plt.ylabel("Pedal value")
plt.title("Accel and brake")
plt.legend()
plt.grid(True)
plt.show()

# 3. Руль и сенсор прямо
plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["steer"], label="steer")
plt.plot(df["step"], df["front"] / 100, label="front / 100")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Steer and front sensor")
plt.legend()
plt.grid(True)
plt.show()


# 4. Проблемная зона 
zone = df[(df["step"] >= 4300) & (df["step"] <= 5000)]

plt.figure(figsize=(14, 6))
plt.plot(zone["step"], zone["speedX"], label="speedX")
plt.plot(zone["step"], zone["target_speed"], label="target_speed")
plt.plot(zone["step"], zone["accel"] * 100, label="accel x100")
plt.plot(zone["step"], zone["brake"] * 100, label="brake x100")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Zoom: problem zone")
plt.legend()
plt.grid(True)
plt.show()


# 5. Положение на трассе
plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["trackPos"], label="trackPos")
plt.axhline(1.0, linestyle="--", label="right edge")
plt.axhline(-1.0, linestyle="--", label="left edge")
plt.xlabel("Step")
plt.ylabel("trackPos")
plt.title("Track position")
plt.legend()
plt.grid(True)
plt.show()

# 6. Анализ Апексов

plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["trackPos"], label="trackPos")
plt.plot(df["step"], df["apex_target"], label="apex_target")
plt.axhline(1.0, linestyle="--", label="right edge")
plt.axhline(-1.0, linestyle="--", label="left edge")
plt.xlabel("Step")
plt.ylabel("Position")
plt.title("Track position vs apex target")
plt.legend()
plt.grid(True)
plt.show()