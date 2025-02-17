import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def set_fonts():
    # Set global font and style parameters for Matplotlib
    plt.rc("text", usetex=False)
    plt.rc("font", size=14, family="serif")
    plt.rc("axes", titlesize=14, labelsize=14)
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    plt.rc("legend", fontsize=13)
    plt.rc("figure", titlesize=17, dpi=150, figsize=(7, 5))
    plt.rc("axes", xmargin=0)
    plt.rc("animation", html="html5")

set_fonts()

def rotation_matrix(roll, pitch, yaw):
    """
    Compute the rotation matrix given roll, pitch, and yaw.
    (Convention: R = R_z(yaw) @ R_y(pitch) @ R_x(roll))
    """
    R_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def visualize_trajectory_3d(xc, uc, N, T_max, cmap='viridis'):
    """
    Visualize the rocket landing trajectory in 3D.
    
    Parameters:
      xc : np.array, shape=(9, N)
           State vector (each row represents):
             0: x (downrange;)
             1: y (crossrange;)
             2: z (altitude;)
             3: vx
             4: vy
             5: vz
             6: roll
             7: pitch
             8: yaw
      uc : np.array, shape=(3, N)
           Control input (ux, uy, uz)
           → In the body frame, thrust is represented as [ux, uy, uz]
      N  : Number of frames
      T_max : Maximum thrust (for scaling purposes)
      cmap  : Colormap name (to display speed-dependent color)
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute speed magnitude using vx, vy, vz
    vct = np.linalg.norm(xc[3:6, :], axis=0)
    norm = Normalize(vmin=vct.min(), vmax=vct.max())
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Create the 3D trajectory by grouping consecutive points (using groups of 4 points)
    segments = [np.column_stack([xc[0, k:k+3],
                                 xc[1, k:k+3],
                                 xc[2, k:k+3]]) for k in range(N-2)]
    line_colors = [scalar_map.to_rgba(vct[k]) for k in range(N-2)]
    traj_collection = Line3DCollection(segments, colors=line_colors, linewidths=2)
    ax.add_collection3d(traj_collection)
    
    # Plot the Lorentz cone (second-order cone, SOC) with a 45° slope.
    # The cone is defined by: z = sqrt(x^2 + y^2)
    span = abs(xc[2, 0])
    CZ = np.linspace(0, span, 50)    
    theta = np.linspace(0, 2 * np.pi, 50)
    THETA, CZ = np.meshgrid(theta, CZ)
    CX = np.tan(np.deg2rad(45)) * CZ * np.cos(THETA)
    CY = np.tan(np.deg2rad(45)) * CZ * np.sin(THETA)
    ax.plot_surface(CX, CY, CZ, color='slategrey', alpha=0.1, rstride=1, cstride=1, 
                    linewidth=0, antialiased=True)
    
    ax.set_xlabel("Downrange [m] (x)")
    ax.set_ylabel("Crossrange [m] (y)")
    ax.set_zlabel("Altitude [m] (z)")
    ax.grid(True)
    
    # Define the rocket model as a rectangular box in the body frame.
    width = 0.2
    depth = 0.2
    height = 1.0
    # Rocket vertices in the body frame (base at z=0, top at z=height)
    rocket_body_vertices = np.array([
        [-width/2, -depth/2, -height/2],
        [ width/2, -depth/2, -height/2],
        [ width/2,  depth/2, -height/2],
        [-width/2,  depth/2, -height/2],
        [-width/2, -depth/2, height/2],
        [ width/2, -depth/2, height/2],
        [ width/2,  depth/2, height/2],
        [-width/2,  depth/2, height/2]
    ])
    # Define the faces of the box by the indices of the vertices
    faces_idx = [
        [0, 1, 2, 3],  # bottom face
        [4, 5, 6, 7],  # top face (rocket's top)
        [0, 1, 5, 4],  # side faces
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]
    
    # Create the initial rocket model by transforming body frame vertices into world coordinates
    pos0 = np.array([xc[0, 0], xc[1, 0], xc[2, 0]])
    roll0, pitch0, yaw0 = xc[6, 0], xc[7, 0], xc[8, 0]
    R0 = rotation_matrix(roll0, pitch0, yaw0)
    rocket_transformed = (R0 @ rocket_body_vertices.T).T + pos0
    faces = [rocket_transformed[face] for face in faces_idx]

    # Create a Poly3DCollection for the rocket model
    rocket_poly = Poly3DCollection(faces, facecolors='darkslategrey', edgecolor='k', alpha=0.9)
    ax.add_collection3d(rocket_poly)
    
    # Create thrust (tomato) arrows using quiver
    thrust_pos0 = (R0 @ np.array([0, 0, -height/2]).T).T + pos0
    thrust_arrow = ax.quiver(thrust_pos0[0], thrust_pos0[1], thrust_pos0[2],
                             0, 0, 0, color='tomato', length=2, arrow_length_ratio=0.3)
    
    def animate(i):
        nonlocal rocket_poly, thrust_arrow
        # Get the current position and orientation
        pos = np.array([xc[0, i], xc[1, i], xc[2, i]])
        roll_i = xc[6, i]
        pitch_i = xc[7, i]
        yaw_i = xc[8, i]
        R = rotation_matrix(roll_i, pitch_i, yaw_i)
        
        # Transform the rocket body vertices to world coordinates for the current frame
        rocket_transformed = (R @ rocket_body_vertices.T).T + pos
        new_faces = [rocket_transformed[face] for face in faces_idx]
        
        # Remove the previous Poly3DCollection and add a new one to avoid edgecolor issues
        rocket_poly.remove()
        rocket_poly = Poly3DCollection(new_faces, facecolors='darkslategrey', edgecolor='k', alpha=0.9)
        ax.add_collection3d(rocket_poly)
                
        # Convert control input (thrust) from the body frame ([ux, uy, uz]) to world coordinates
        thrust_world = R @ uc[:,i]
        thrust_norm = np.linalg.norm(thrust_world)
        if thrust_norm > 0:
            thrust_dir = thrust_world / thrust_norm
        else:
            thrust_dir = thrust_world
        thrust_arrow.remove()    
        thrust_pos = (R @ np.array([0, 0, -height/2]).T).T + pos
        thrust_arrow = ax.quiver(thrust_pos[0], thrust_pos[1], thrust_pos[2],
                                 -thrust_dir[0], -thrust_dir[1], -thrust_dir[2],
                                 color='tomato', length=thrust_norm / T_max, arrow_length_ratio=0.3)
        return rocket_poly, thrust_arrow

    anim = animation.FuncAnimation(fig, animate, frames=N, interval=60, repeat=False, blit=False)
    
    # Add a colorbar for the speedqq
    cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.5, aspect=40, label="Speed [m/s]")
    
    # Create custom legend handles (proxy artists) to avoid issues with 3D objects in the legend
    rocket_handle = Patch(facecolor='darkslategrey', edgecolor='k', label='Rocket')
    glide_handle = Line2D([0], [0], color='slategrey', linestyle='dashed', label='Glide-slope')
    thrust_handle = Line2D([0], [0], color='tomato', label='Thrust')
    
    ax.legend(handles=[rocket_handle, glide_handle, thrust_handle],
              loc="upper left", fontsize=12)
    
    ax.set_title("Rocket Landing Trajectory (3D)", pad=20)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, 10)

    ax.view_init(elev=90, azim=45)
    anim.save("gif_name.gif", writer="pillow", fps=20, dpi=150)
    
    plt.show()

file_path = 'file_path.xls'
df = pd.read_excel(file_path, sheet_name='sheet_name')


x = df['RX'].values
y = df['RY'].values
z = df['RZ'].values

vx = df['VX'].values
vy = df['VY'].values
vz = df['VZ'].values

q0 = df['Q0'].values
q1 = df['Q1'].values
q2 = df['Q2'].values
q3 = df['Q3'].values

roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

xc = np.vstack([x, y, z, vx, vy, vz, roll, pitch, yaw])

# Control input: ux, uy, uz (body frame: [ux, uy, uz])
ux = df['UX'].values
uy = df['UY'].values
uz = df['UZ'].values
uc = np.vstack([ux, uy, uz])

N = len(ux)
T_max = 1.1 * 10 * 9.81

visualize_trajectory_3d(xc, uc, N, T_max)
