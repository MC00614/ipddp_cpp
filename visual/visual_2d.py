import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def set_fonts():
    plt.rc("text")
    plt.rc("font", size=14, family="serif")
    plt.rc("axes", titlesize=14, labelsize=14)
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    plt.rc("legend", fontsize=13)
    plt.rc("figure", titlesize=17, dpi=150, figsize=(7, 5))
    plt.rc("axes", xmargin=0)
    plt.rc("animation", html="html5")

set_fonts()

def visualize_trajectory(xc, uc, N, T_max, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(9, 6))
    vct = np.linalg.norm(xc[2:4, :], axis=0)
    norm = Normalize(vmin=vct.min(), vmax=vct.max())
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    theta = xc[4, :]
    ux = uc[0, :] * np.cos(theta) - uc[1, :] * np.sin(theta)
    uy = uc[0, :] * np.sin(theta) + uc[1, :] * np.cos(theta)
    thrust_magnitude = np.linalg.norm(np.vstack([ux, uy]), axis=0)
    
    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")
    ax.grid(color="0.9")
    
    line_segments = [np.column_stack([xc[0, k:k+4], xc[1, k:k+4]]) for k in range(N-3)]
    line_colors = [scalar_map.to_rgba(vct[k]) for k in range(N-3)]
    trajectory = LineCollection(line_segments, colors=line_colors, linewidths=2)
    trajectory.set_zorder(0)
    ax.add_collection(trajectory)

    span_gs = max(100, 2.25 * abs(xc[0, 0]))
    x_cone = np.array([-span_gs, 0, span_gs])
    y_cone = np.tan(np.radians(45)) * np.abs(x_cone)
    ax.plot(x_cone, y_cone, linestyle='dashed', linewidth=1.5, color='slategrey', alpha=0.75, label='Glide-slope')

    width = 0.2
    height = 1
    rocket_rect = plt.Rectangle((0, 0), width, height, angle=0, color="darkslategrey", label="Rocket")
    ax.add_patch(rocket_rect)

    thrust_quiver = ax.quiver([], [], [], [], color='tomato', pivot='tail', scale= 0.15*T_max, width = 0.004, headwidth=0.004)

    def animate(i):
        rocket_rect.set_xy((xc[0, i] - width/2, xc[1, i] - height/2))
        rocket_rect.angle = np.degrees(xc[4, i])
        
        try:
            thrust_quiver.set_offsets([xc[0, i], xc[1, i] - height/2])
            thrust_quiver.set_UVC(-ux[i] / thrust_magnitude[i], -uy[i] / thrust_magnitude[i])
        except:
            pass

        return rocket_rect, thrust_quiver
    
    anim = animation.FuncAnimation(fig, animate, frames=N, interval=60)
    
    cbar = fig.colorbar(scalar_map, ax=ax, aspect=40, label="Speed [m/s]")
    ax.legend(fontsize=12, loc="upper left")
    ax.set_title("Rocket-landing trajectory", fontsize=13, pad=12.5)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-1, 12)
    
    anim.save("gif_name.gif", writer="pillow", fps=20)
    
    plt.show()
    
import pandas as pd
file_path = 'file_path.xls'
df = pd.read_excel(file_path, sheet_name='sheet_name')
Y = df['Y'].values
X = df['X'].values
VY = df['VY'].values
VX = df['VX'].values
W = - df['W'].values
A = - df['A'].values
UY = df['UY'].values
UX = df['UX'].values
    
# Example Usage
N = len(UY)
xc = np.vstack([X, Y, VX, VY, W, A])
uc = np.vstack([UX, UY])
T_max = 1.1*10*9.81
visualize_trajectory(xc, uc, N, T_max)
